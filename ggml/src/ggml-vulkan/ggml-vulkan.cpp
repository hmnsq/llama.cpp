#ಅನ್ತರ್ಭಾವ "ggml-vulkan-common.h"

ಸ್ಥಿರ vk_device_architecture get_device_architecture(ಸ್ಥಿರಾಂಕ vk::PhysicalDevice& device) {
    vk::PhysicalDeviceProperties props = device.getProperties();

    ಆದರೆ (props.vendorID == VK_VENDOR_ID_AMD) {
        ಸ್ಥಿರಾಂಕ std::vector<vk::ExtensionProperties> ext_props = device.enumerateDeviceExtensionProperties();

        ತಾರ್ಕಿಕ amd_shader_core_properties = ಸುಳ್ಳು;
        ತಾರ್ಕಿಕ integer_dot_ಗುಣಲಬ್ಧ = ಸುಳ್ಳು;
        ತಾರ್ಕಿಕ subgroup_ಗಾತ್ರ_control = ಸುಳ್ಳು;
        ತಾರ್ಕಿಕ shader_float8 = ಸುಳ್ಳು;

        ಗಾಗಿ (ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ& properties : ext_props) {
            ಆದರೆ (strcmp("VK_AMD_shader_core_properties", properties.extensionName) == 0) {
                amd_shader_core_properties = ನಿಜ;
            } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (strcmp("VK_KHR_shader_integer_dot_product", properties.extensionName) == 0) {
                integer_dot_ಗುಣಲಬ್ಧ = ನಿಜ;
            } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (strcmp("VK_EXT_subgroup_size_control", properties.extensionName) == 0) {
                subgroup_ಗಾತ್ರ_control = ನಿಜ;
            } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (strcmp("VK_EXT_shader_float8", properties.extensionName) == 0) {
                shader_float8 = ನಿಜ;
            }
        }

        ಆದರೆ (!amd_shader_core_properties || !integer_dot_ಗುಣಲಬ್ಧ || !subgroup_ಗಾತ್ರ_control) {
            ಹಿಂತಿರುಗಿ vk_device_architecture::OTHER;
        }

        vk::PhysicalDeviceProperties2 props2;
        vk::PhysicalDeviceShaderCorePropertiesAMD shader_core_props_amd;
        vk::Physical_Device_Shader_Integer_Dot_ಗುಣಲಬ್ಧ_Properties_KHR integer_dot_props;
        vk::Physical_Device_Subgroup_ಗಾತ್ರ_Control_Properties_EXT subgroup_ಗಾತ್ರ_control_props;

        props2.p_ಮುಂದಿನ = &shader_core_props_amd;
        shader_core_props_amd.p_ಮುಂದಿನ = &integer_dot_props;
        integer_dot_props.p_ಮುಂದಿನ = &subgroup_ಗಾತ್ರ_control_props;

        device.getProperties2(&props2);

        ಆದರೆ (subgroup_ಗಾತ್ರ_control_props.max_Subgroup_ಗಾತ್ರ == 64 && subgroup_ಗಾತ್ರ_control_props.min_Subgroup_ಗಾತ್ರ == 64) {
            ಹಿಂತಿರುಗಿ vk_device_architecture::AMD_GCN;
        }
        ಆದರೆ (subgroup_ಗಾತ್ರ_control_props.max_Subgroup_ಗಾತ್ರ == 64 && subgroup_ಗಾತ್ರ_control_props.min_Subgroup_ಗಾತ್ರ == 32) {
            // RDNA
            ಆದರೆ (shader_core_props_amd.wavefronts_ಪ್ರತಿ_Simd == 20) {
                ಹಿಂತಿರುಗಿ vk_device_architecture::AMD_RDNA1;
            }
            ಆದರೆ (shader_float8) {
                ಹಿಂತಿರುಗಿ vk_device_architecture::AMD_RDNA4;
            }
            ಆದರೆ (integer_dot_props.integer_Dot_Product4x8_ಬಿಟ್_Packed_Mixed_Signedness_Accelerated) {
                ಹಿಂತಿರುಗಿ vk_device_architecture::AMD_RDNA3;
            }
            ಹಿಂತಿರುಗಿ vk_device_architecture::AMD_RDNA2;
        }
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (props.vendorID == VK_VENDOR_ID_INTEL) {
        ಸ್ಥಿರಾಂಕ std::vector<vk::ExtensionProperties> ext_props = device.enumerateDeviceExtensionProperties();

        ತಾರ್ಕಿಕ subgroup_ಗಾತ್ರ_control = ಸುಳ್ಳು;
        ತಾರ್ಕಿಕ integer_dot_ಗುಣಲಬ್ಧ = ಸುಳ್ಳು;

        ಗಾಗಿ (ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ& properties : ext_props) {
            ಆದರೆ (strcmp("VK_EXT_subgroup_size_control", properties.extensionName) == 0) {
                subgroup_ಗಾತ್ರ_control = ನಿಜ;
            } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (strcmp("VK_KHR_shader_integer_dot_product", properties.extensionName) == 0) {
                integer_dot_ಗುಣಲಬ್ಧ = ನಿಜ;
            }
        }

        ಆದರೆ (!subgroup_ಗಾತ್ರ_control || !integer_dot_ಗುಣಲಬ್ಧ) {
            ಹಿಂತಿರುಗಿ vk_device_architecture::OTHER;
        }

        vk::PhysicalDeviceProperties2 props2;
        vk::Physical_Device_Subgroup_ಗಾತ್ರ_Control_Properties_EXT subgroup_ಗಾತ್ರ_control_props;
        vk::Physical_Device_Shader_Integer_Dot_ಗುಣಲಬ್ಧ_Properties_KHR integer_dot_props;

        props2.p_ಮುಂದಿನ = &subgroup_ಗಾತ್ರ_control_props;
        subgroup_ಗಾತ್ರ_control_props.p_ಮುಂದಿನ = &integer_dot_props;
        device.getProperties2(&props2);

        ಆದರೆ (subgroup_ಗಾತ್ರ_control_props.min_Subgroup_ಗಾತ್ರ == 16) {
            // Xe2 architecture uses SIMD16 while previous Xe and Gen architecture uses SIMD8.
            // Minimum subgroup size matches the SIMD width so we distinguish architecture by checking this value.
            // https://www.intel.com/content/www/us/en/content-details/824434/2024-intel-tech-tour-xe2-and-lunar-lake-s-gpu.html
            // https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2025-0/intel-xe-gpu-architecture.html
            ಹಿಂತಿರುಗಿ vk_device_architecture::INTEL_XE2;
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (subgroup_ಗಾತ್ರ_control_props.min_Subgroup_ಗಾತ್ರ == 8 &&
                 integer_dot_ಗುಣಲಬ್ಧ && integer_dot_props.integer_Dot_Product4x8_ಬಿಟ್_Packed_Signed_Accelerated) {
            ಹಿಂತಿರುಗಿ vk_device_architecture::INTEL_XE1;
        }
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (props.vendorID == VK_VENDOR_ID_NVIDIA) {
        ಸ್ಥಿರಾಂಕ std::vector<vk::ExtensionProperties> ext_props = device.enumerateDeviceExtensionProperties();

        ತಾರ್ಕಿಕ cooperative_ಮಾತೃಕೆ = ಸುಳ್ಳು;
        ತಾರ್ಕಿಕ sm_builtins = ಸುಳ್ಳು;

        // Detect "pre-turing" based on lack of coopmat support.
        ಗಾಗಿ (ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ& properties : ext_props) {
            ಆದರೆ (strcmp("VK_KHR_cooperative_matrix", properties.extensionName) == 0) {
                cooperative_ಮಾತೃಕೆ = ನಿಜ;
            } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (strcmp("VK_NV_shader_sm_builtins", properties.extensionName) == 0) {
                sm_builtins = ನಿಜ;
            }
        }

        ಆದರೆ (!cooperative_ಮಾತೃಕೆ) {
            ಹಿಂತಿರುಗಿ vk_device_architecture::NVIDIA_PRE_TURING;
        }

        ಆದರೆ (sm_builtins) {
            vk::PhysicalDeviceProperties2 props2;
            vk::PhysicalDeviceShaderSMBuiltinsPropertiesNV sm_props;

            props2.p_ಮುಂದಿನ = &sm_props;

            device.getProperties2(&props2);

            // Turing has 32, following architectures have 48
            ಆದರೆ (sm_props.shader_Warps_ಪ್ರತಿ_SM == 32) {
                ಹಿಂತಿರುಗಿ vk_device_architecture::NVIDIA_TURING;
            }
        }
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ(props.vendorID == VK_VENDOR_ID_QUALCOMM){
        ಸ್ಥಿರಾಂಕ std::vector<vk::ExtensionProperties> ext_props = device.enumerateDeviceExtensionProperties();

        ತಾರ್ಕಿಕ cooperative_ಮಾತೃಕೆ = ಸುಳ್ಳು;
        ತಾರ್ಕಿಕ cooperative_ಮಾತೃಕೆ_conversion = ಸುಳ್ಳು;

        ಗಾಗಿ (ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ& properties : ext_props) {
            ಆದರೆ (strcmp("VK_KHR_cooperative_matrix", properties.extensionName) == 0) {
                cooperative_ಮಾತೃಕೆ = ನಿಜ;
            } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (strcmp("VK_QCOM_cooperative_matrix_conversion", properties.extensionName) == 0) {
                cooperative_ಮಾತೃಕೆ_conversion = ನಿಜ;
            }
        }

        ಆದರೆ (cooperative_ಮಾತೃಕೆ && cooperative_ಮಾತೃಕೆ_conversion) {
            ಹಿಂತಿರುಗಿ vk_device_architecture::QUALCOMM_ADRENO;
        }
    }
    ಹಿಂತಿರುಗಿ vk_device_architecture::OTHER;
}

ತಾರ್ಕಿಕ ggml_vk_lightning_indexer_k_type_supported(ggml_type type) {
    ಹಿಂತಿರುಗಿ std::find(lightning_indexer_k_types.begin(), lightning_indexer_k_types.end(), type) != lightning_indexer_k_types.end();
}
ಶೂನ್ಯ ggml_vk_print_device_fault_info(ಸ್ಥಿರಾಂಕ vk_device& device) {
    ಆದರೆ (!device->device_fault || !device->pfn_vkGetDeviceFaultInfoEXT) {
        ಹಿಂತಿರುಗಿ;
    }

    VkDeviceFaultCountsEXT fault_counts {};
    fault_counts.sType = VK_STRUCTURE_TYPE_DEVICE_FAULT_COUNTS_EXT;
    Vk_ಫಲಿತಾಂಶ res = device->pfn_vkGetDeviceFaultInfoEXT(device->device, &fault_counts, ಶೂನ್ಯಸೂಚಕ);
    ಆದರೆ (res != VK_SUCCESS) {
        GGML_LOG_ERROR("ggml_vulkan: vkGetDeviceFaultInfoEXT (counts) failed: %d\n", res);
        ಹಿಂತಿರುಗಿ;
    }

    std::vector<Vk_Device_Fault_ವಿಳಾಸ_Info_EXT> ವಿಳಾಸ_infos(fault_counts.ವಿಳಾಸ_Info_ಎಣಿಕೆ);
    std::vector<VkDeviceFaultVendorInfoEXT> vendor_infos(fault_counts.vendor_Info_ಎಣಿಕೆ);

    VkDeviceFaultInfoEXT fault_info {};
    fault_info.sType = VK_STRUCTURE_TYPE_DEVICE_FAULT_INFO_EXT;
    fault_info.p_ವಿಳಾಸ_Infos = ವಿಳಾಸ_infos.ದತ್ತಾಂಶ();
    fault_info.pVendorInfos = vendor_infos.ದತ್ತಾಂಶ();

    res = device->pfn_vkGetDeviceFaultInfoEXT(device->device, &fault_counts, &fault_info);
    ಆದರೆ (res != VK_SUCCESS) {
        GGML_LOG_ERROR("ggml_vulkan: vkGetDeviceFaultInfoEXT (info) failed: %d\n", res);
        ಹಿಂತಿರುಗಿ;
    }

    ಆದರೆ (fault_counts.ವಿಳಾಸ_Info_ಎಣಿಕೆ == 0 && fault_counts.vendor_Info_ಎಣಿಕೆ == 0 && fault_info.description[0] == '\0') {
        ಹಿಂತಿರುಗಿ;
    }

    ಆದರೆ (fault_info.description[0] != '\0') {
        GGML_LOG_ERROR("ggml_vulkan: device fault on %s: %s\n", device->name.c_str(), fault_info.description);
    }

    ಗಾಗಿ (uint32_t i = 0; i < fault_counts.ವಿಳಾಸ_Info_ಎಣಿಕೆ; i++) {
        ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ& info = ವಿಳಾಸ_infos[i];
        GGML_LOG_CONT("  address fault %u: type=%d address=0x%llx precision=0x%llx\n",
                i, (ಪೂರ್ಣಾಂಕ)info.ವಿಳಾಸ_Type,
                (ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ ದೀರ್ಘ)info.reported_ವಿಳಾಸ,
                (ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ ದೀರ್ಘ)info.ವಿಳಾಸ_Precision);
    }
    ಗಾಗಿ (uint32_t i = 0; i < fault_counts.vendor_Info_ಎಣಿಕೆ; i++) {
        ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ& info = vendor_infos[i];
        GGML_LOG_CONT("  vendor fault %u: %s (code=0x%llx data=0x%llx)\n",
                i, info.description,
                (ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ ದೀರ್ಘ)info.vendor_Fault_ಸಂಕೇತ,
                (ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ ದೀರ್ಘ)info.vendor_Fault_ದತ್ತಾಂಶ);
    }
}
uint64_t ggml_vk_get_ಗಂಟು_flops(ಸ್ಥಿರಾಂಕ ggml_tensor * ಗಂಟು) {
    ಆದರೆ (ಗಂಟು->op == GGML_OP_MUL_MAT || ಗಂಟು->op == GGML_OP_MUL_MAT_ID) {
        ಸ್ಥಿರಾಂಕ uint64_t m     = ಗಂಟು->ne[0];
        ಸ್ಥಿರಾಂಕ uint64_t n     = ಗಂಟು->ne[1];
        ಸ್ಥಿರಾಂಕ uint64_t k     = ಗಂಟು->src[1]->ne[0];
        ಸ್ಥಿರಾಂಕ uint64_t batch = ಗಂಟು->ne[2] * ಗಂಟು->ne[3];
        ಹಿಂತಿರುಗಿ m * n * (k + (k - 1)) * batch;
    }
    ಆದರೆ (ಗಂಟು->op == GGML_OP_CONV_2D || ಗಂಟು->op == GGML_OP_CONV_TRANSPOSE_2D) {
        ಸ್ಥಿರಾಂಕ ggml_tensor * knl = ಗಂಟು->src[0];
        ಸ್ಥಿರಾಂಕ uint64_t Cout  = ಗಂಟು->ne[2];
        ಸ್ಥಿರಾಂಕ uint64_t ಗಾತ್ರ_K = ಗಂಟು->src[1]->ne[2] * knl->ne[0] * knl->ne[1];
        ಸ್ಥಿರಾಂಕ uint64_t ಗಾತ್ರ_N_2 = ಗಂಟು->ne[3] * ಗಂಟು->ne[0] * ಗಂಟು->ne[1];
        ಹಿಂತಿರುಗಿ Cout * ಗಾತ್ರ_N_2 * (ಗಾತ್ರ_K + (ಗಾತ್ರ_K - 1));
    }
    ಆದರೆ (ಗಂಟು->op == GGML_OP_CONV_3D) {
        ಸ್ಥಿರಾಂಕ ggml_tensor * knl = ಗಂಟು->src[0];
        ಸ್ಥಿರಾಂಕ uint64_t OC     = ggml_get_op_params_i32(ಗಂಟು, 11);
        ಸ್ಥಿರಾಂಕ uint64_t IC     = ggml_get_op_params_i32(ಗಂಟು, 9);
        ಸ್ಥಿರಾಂಕ uint64_t ಗಾತ್ರ_K = IC * knl->ne[0] * knl->ne[1] * knl->ne[2];
        ಸ್ಥಿರಾಂಕ uint64_t ಗಾತ್ರ_N_2 = ಗಂಟು->ne[3] / OC * ಗಂಟು->ne[0] * ಗಂಟು->ne[1] * ಗಂಟು->ne[2];
        ಹಿಂತಿರುಗಿ OC * ಗಾತ್ರ_N_2 * (ಗಾತ್ರ_K + (ಗಾತ್ರ_K - 1));
    }
    ಆದರೆ (ಗಂಟು->op == GGML_OP_FLASH_ATTN_EXT) {
        ಸ್ಥಿರಾಂಕ ggml_tensor * q = ಗಂಟು->src[0];
        ಸ್ಥಿರಾಂಕ ggml_tensor * k = ಗಂಟು->src[1];
        ಸ್ಥಿರಾಂಕ ggml_tensor * v = ಗಂಟು->src[2];
        ಹಿಂತಿರುಗಿ 2ull * q->ne[1] * q->ne[2] * (k->ne[0] + v->ne[0]) * k->ne[1] * q->ne[3];
    }
    ಹಿಂತಿರುಗಿ 0;
}
ಶೂನ್ಯ ggml_vk_print_ಗಂಟು_list(ಸ್ಥಿರಾಂಕ ggml_cgraph * cgraph, ಪೂರ್ಣಾಂಕ start, ಪೂರ್ಣಾಂಕ end) {
    uint64_t ಒಟ್ಟು_flops = 0;
    ಪೂರ್ಣಾಂಕ n_ops = 0;
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = start; j <= end && j < cgraph->n_nodes; j++) {
        uint64_t flops = ggml_vk_get_ಗಂಟು_flops(cgraph->nodes[j]);
        ಒಟ್ಟು_flops += flops;
        n_ops++;
        ಆದರೆ (flops > 0) {
            GGML_LOG_CONT("  node %d: %s (%s) [%.2f GFLOP]\n",
                    j, cgraph->nodes[j]->name, ggml_op_name(cgraph->nodes[j]->op),
                    flops / 1e9);
        } ಇಲ್ಲದಿದ್ದರೆ {
            GGML_LOG_CONT("  node %d: %s (%s)\n",
                    j, cgraph->nodes[j]->name, ggml_op_name(cgraph->nodes[j]->op));
        }
    }
    GGML_LOG_CONT("  total: %d ops, %.2f GFLOP\n", n_ops, ಒಟ್ಟು_flops / 1e9);
}
ಶೂನ್ಯ ggml_vk_print_device_ಕಳೆದುಹೋದ_info(ಸ್ಥಿರಾಂಕ vk_device& device) {
    ggml_vk_print_device_fault_info(device);
    ಆದರೆ (device->serialize_submissions && device->diag_cgraph != ಶೂನ್ಯಸೂಚಕ && device->diag_prev_ಆರಂಭ >= 0) {
        GGML_LOG_ERROR("ggml_vulkan: device lost on %s, likely caused by previous submission (nodes %d to %d):\n",
                device->name.c_str(), device->diag_prev_ಆರಂಭ, device->diag_prev_ಕೊನೆ);
        ggml_vk_print_ಗಂಟು_list(device->diag_cgraph, device->diag_prev_ಆರಂಭ, device->diag_prev_ಕೊನೆ);
    } ಇಲ್ಲದಿದ್ದರೆ {
        GGML_LOG_ERROR("ggml_vulkan: device lost on %s\n", device->name.c_str());
    }
}
ಶೂನ್ಯ * ಸ್ಥಿರಾಂಕ vk_ptr_ತಳ = (ಶೂನ್ಯ *)(uintptr_t) 0x1000;  // NOLINT

uint64_t vk_tensor_ಸ್ಥಾನಾಂತರ(ಸ್ಥಿರಾಂಕ ggml_tensor * tensor) {
    ಆದರೆ (tensor->view_src) {
        ಹಿಂತಿರುಗಿ (uint8_t *) tensor->view_src->ದತ್ತಾಂಶ - (uint8_t *) vk_ptr_ತಳ;
    }
    ಹಿಂತಿರುಗಿ (uint8_t *) tensor->ದತ್ತಾಂಶ - (uint8_t *) vk_ptr_ತಳ;
}

size_t ggml_vk_tensor_ಬಫರ್_ಸ್ಥಾನಾಂತರ(ಸ್ಥಿರಾಂಕ ggml_backend_vk_context * ctx, ಸ್ಥಿರಾಂಕ ggml_tensor * t) {
    // vk_tensor_offset() is relative to vk_ptr_base, but mapped host tensors need an offset relative to their Vulkan buffer.
    ಆದರೆ (ctx->device->uma) {
        vk_ಬಫರ್ buf = ಶೂನ್ಯಸೂಚಕ;
        size_t off = 0;
        ggml_vk_host_get(ctx->device, t->ದತ್ತಾಂಶ, buf, off);
        ಆದರೆ (buf) {
            ಹಿಂತಿರುಗಿ off;
        }
    }
    ಹಿಂತಿರುಗಿ (size_t)(vk_tensor_ಸ್ಥಾನಾಂತರ(t) + t->view_offs);
}
size_t ggml_vk_descriptor_ಸ್ಥಾನಾಂತರ(size_t tensor_ಸ್ಥಾನಾಂತರ, size_t alignment, size_t type_ಗಾತ್ರ) {
    // Move the descriptor back until its distance to the tensor is divisible by the tensor type size.
    size_t descriptor_ಸ್ಥಾನಾಂತರ = tensor_ಸ್ಥಾನಾಂತರ & ~(alignment - 1);
    ಆಗ ((tensor_ಸ್ಥಾನಾಂತರ - descriptor_ಸ್ಥಾನಾಂತರ) % type_ಗಾತ್ರ != 0) {
        GGML_ASSERT(descriptor_ಸ್ಥಾನಾಂತರ >= alignment);
        descriptor_ಸ್ಥಾನಾಂತರ -= alignment;
    }

    ಹಿಂತಿರುಗಿ descriptor_ಸ್ಥಾನಾಂತರ;
}
uint32_t get_misalign_ಬೈಟುಗಳು(ಸ್ಥಿರಾಂಕ ggml_backend_vk_context * ctx, ಸ್ಥಿರಾಂಕ ggml_tensor * t) {
    ಸ್ಥಿರಾಂಕ size_t tensor_ಸ್ಥಾನಾಂತರ = ggml_vk_tensor_ಬಫರ್_ಸ್ಥಾನಾಂತರ(ctx, t);
    ಸ್ಥಿರಾಂಕ size_t descriptor_ಸ್ಥಾನಾಂತರ = ggml_vk_descriptor_ಸ್ಥಾನಾಂತರ(
        tensor_ಸ್ಥಾನಾಂತರ, ctx->device->properties.limits.min_Storage_ಬಫರ್_ಸ್ಥಾನಾಂತರ_Alignment, ggml_type_ಗಾತ್ರ(t->type));
    GGML_ASSERT(tensor_ಸ್ಥಾನಾಂತರ - descriptor_ಸ್ಥಾನಾಂತರ <= UINT32_MAX);
    ಹಿಂತಿರುಗಿ tensor_ಸ್ಥಾನಾಂತರ - descriptor_ಸ್ಥಾನಾಂತರ;
}

uint32_t ggml_vk_concat_unit_ಗಾತ್ರ(ggml_type type) {
    ಸ್ಥಿರಾಂಕ uint32_t type_ಗಾತ್ರ = ggml_type_ಗಾತ್ರ(type);

    ಆದರೆ (!ggml_is_quantized(type)) {
        ಹಿಂತಿರುಗಿ type_ಗಾತ್ರ;
    }

    // Use the widest existing concat shader that evenly divides a quant block.
    ಆದರೆ (type_ಗಾತ್ರ % 8 == 0) {
        ಹಿಂತಿರುಗಿ 8;
    }
    ಆದರೆ (type_ಗಾತ್ರ % 4 == 0) {
        ಹಿಂತಿರುಗಿ 4;
    }
    ಆದರೆ (type_ಗಾತ್ರ % 2 == 0) {
        ಹಿಂತಿರುಗಿ 2;
    }
    ಹಿಂತಿರುಗಿ 1;
}
ತಾರ್ಕಿಕ ggml_vk_concat_supported(ಸ್ಥಿರಾಂಕ ggml_tensor * src0, ಸ್ಥಿರಾಂಕ ggml_tensor * src1, ಸ್ಥಿರಾಂಕ ggml_tensor * dst) {
    ಆದರೆ (src0->type != src1->type || src0->type != dst->type) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಆದರೆ (!ggml_is_quantized(src0->type)) {
        ಸ್ಥಿರಾಂಕ size_t type_ಗಾತ್ರ = ggml_type_ಗಾತ್ರ(src0->type);
        ಹಿಂತಿರುಗಿ type_ಗಾತ್ರ == 1 || type_ಗಾತ್ರ == 2 || type_ಗಾತ್ರ == 4 || type_ಗಾತ್ರ == 8;
    }

    // Quantized tensor rows are block-aligned when created.
    ಹಿಂತಿರುಗಿ ggml_is_contiguous_ಅಡ್ಡಸಾಲುಗಳು(src0) && ggml_is_contiguous_ಅಡ್ಡಸಾಲುಗಳು(src1) && ggml_is_contiguous_ಅಡ್ಡಸಾಲುಗಳು(dst);
}
ಸ್ಥಿರ ತಾರ್ಕಿಕ vk_instance_initialized = ಸುಳ್ಳು;

vk_instance_t vk_instance;

ಸ್ಥಿರ Vk_Device_ಗಾತ್ರ ggml_vk_get_max_ಬಫರ್_range(ಸ್ಥಿರಾಂಕ ggml_backend_vk_context * ctx, ಸ್ಥಿರಾಂಕ vk_ಬಫರ್ &buf, ಸ್ಥಿರಾಂಕ Vk_Device_ಗಾತ್ರ ಸ್ಥಾನಾಂತರ) {
    ಸ್ಥಿರಾಂಕ Vk_Device_ಗಾತ್ರ range = std::min(Vk_Device_ಗಾತ್ರ{buf->size - ಸ್ಥಾನಾಂತರ},
                                        Vk_Device_ಗಾತ್ರ{ctx->device->properties.limits.max_Storage_ಬಫರ್_Range});
    ಹಿಂತಿರುಗಿ range;
}

ಶೂನ್ಯ ggml_vk_wait_for_fence(ggml_backend_vk_context * ctx) {
    // Use waitForFences while most of the graph executes. Hopefully the CPU can sleep
    // during this wait.
    ಆದರೆ (ctx->almost_ready_fence_pending) {
        VK_CHECK(ctx->device->device.waitForFences({ ctx->almost_ready_fence }, ನಿಜ, UINT64_MAX), "almost_ready_fence", ctx->device);
        ctx->device->device.resetFences({ ctx->almost_ready_fence });
        ctx->almost_ready_fence_pending = ಸುಳ್ಳು;
    }

    // Spin (w/pause) waiting for the graph to finish executing.
    vk::ಫಲಿತಾಂಶ_2 ಫಲಿತಾಂಶ;
    ಗಾಗಿ (;;) {
        ಪ್ರಯತ್ನಿಸು {
            ಫಲಿತಾಂಶ = ctx->device->device.get_Fence_ಸ್ಥಿತಿ(ctx->fence);
        } ಹಿಡಿ (vk::Device_ಕಳೆದುಹೋದ_Error &) {
            ggml_vk_print_device_ಕಳೆದುಹೋದ_info(ctx->device);
            GGML_LOG_ERROR("ggml_vulkan: getFenceStatus at %s:%d\n", __FILE__, __LINE__);
            ಎಸೆ;
        }
        ಆದರೆ (ಫಲಿತಾಂಶ == vk::ಫಲಿತಾಂಶ_2::eSuccess) {
            ನಿಲ್ಲಿಸು;
        }
        ಆದರೆ (ಫಲಿತಾಂಶ != vk::ಫಲಿತಾಂಶ_2::eNotReady) {
            GGML_LOG_ERROR("ggml_vulkan: error %s at %s:%d\n", to_string(ಫಲಿತಾಂಶ).c_str(), __FILE__, __LINE__);
            ಎಸೆ vk::SystemError(vk::make_error_code(ಫಲಿತಾಂಶ), "ggml_vulkan: getFenceStatus");
        }
        ಗಾಗಿ (uint32_t i = 0; i < 100; ++i) {
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

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_vk_strip_decode_ಸದಿಶ(ಸ್ಥಿರಾಂಕ uint32_t * code, size_t ಶಬ್ದ_ಎಣಿಕೆ, std::vector<uint32_t> & ಹೊರಹೋಗು) {
    ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ k_Decode_ಸದಿಶ_Ext[] = "SPV_NV_cooperative_matrix_decode_vector";

    ಆದರೆ (ಶಬ್ದ_ಎಣಿಕೆ < 5) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ತಾರ್ಕಿಕ ಬಳಸುತ್ತದೆ_decode_ಸದಿಶ = ಸುಳ್ಳು;
    ಗಾಗಿ (size_t pos = 5; pos < ಶಬ್ದ_ಎಣಿಕೆ; ) {
        uint32_t ಶಬ್ದ = code[pos];
        uint32_t wc   = ಶಬ್ದ >> spv::ಶಬ್ದ_ಎಣಿಕೆ_ಸರಿಸು;
        uint32_t op   = ಶಬ್ದ & spv::Op_ಸಂಕೇತ_ಮುಖವಾಡ;
        GGML_ASSERT(wc > 0 && pos + wc <= ಶಬ್ದ_ಎಣಿಕೆ);
        ಆದರೆ (op == spv::OpExtension && wc >= 2) {
            ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * s = reinterpret_cast<ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *>(&code[pos + 1]);
            ಆದರೆ (strcmp(s, k_Decode_ಸದಿಶ_Ext) == 0) {
                ಬಳಸುತ್ತದೆ_decode_ಸದಿಶ = ನಿಜ;
                ನಿಲ್ಲಿಸು;
            }
        }
        pos += wc;
    }

    ಆದರೆ (!ಬಳಸುತ್ತದೆ_decode_ಸದಿಶ) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    VK_LOG_DEBUG("ggml_vk_strip_decode_vector: stripping SPV_NV_cooperative_matrix_decode_vector");

    // Bulk-copy unchanged runs and only break the run when an instruction needs to
    // be dropped or patched. Use reserve + insert/push_back so the destination buffer
    // is touched exactly once (no zero-initialization pass from resize()).
    ಹೊರಹೋಗು.clear();
    ಹೊರಹೋಗು.reserve(ಶಬ್ದ_ಎಣಿಕೆ);

    size_t ಸತತ_ಆರಂಭ = 0;
    ಸ್ವಯಂಚಾಲಿತ flush_ಸತತ = [&](size_t ಮೇಲೆ_ಗಮ್ಯ) {
        ಆದರೆ (ಮೇಲೆ_ಗಮ್ಯ > ಸತತ_ಆರಂಭ) {
            ಹೊರಹೋಗು.insert(ಹೊರಹೋಗು.end(), code + ಸತತ_ಆರಂಭ, code + ಮೇಲೆ_ಗಮ್ಯ);
        }
    };

    ಗಾಗಿ (size_t pos = 5; pos < ಶಬ್ದ_ಎಣಿಕೆ; ) {
        uint32_t ಶಬ್ದ = code[pos];
        uint32_t wc   = ಶಬ್ದ >> spv::ಶಬ್ದ_ಎಣಿಕೆ_ಸರಿಸು;
        uint32_t op   = ಶಬ್ದ & spv::Op_ಸಂಕೇತ_ಮುಖವಾಡ;
        GGML_ASSERT(wc > 0 && pos + wc <= ಶಬ್ದ_ಎಣಿಕೆ);

        ಆದರೆ (op == spv::OpExtension && wc >= 2) {
            ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * s = reinterpret_cast<ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *>(&code[pos + 1]);
            ಆದರೆ (strcmp(s, k_Decode_ಸದಿಶ_Ext) == 0) {
                flush_ಸತತ(pos);
                pos += wc;
                ಸತತ_ಆರಂಭ = pos;
                ಮುಂದುವರಿ;
            }
        }

        ಆದರೆ (op == spv::OpCapability && wc == 2 && code[pos + 1] == k_Spv_Capability_Cooperative_ಮಾತೃಕೆ_Decode_ಸದಿಶ_NV) {
            flush_ಸತತ(pos);
            pos += wc;
            ಸತತ_ಆರಂಭ = pos;
            ಮುಂದುವರಿ;
        }

        ಆದರೆ (op == k_Spv_Op_Cooperative_ಮಾತೃಕೆ_ತುಂಬು_Tensor_NV) {
            // [opcode/wc][ResultType][Result][Pointer][Object][TensorLayout][MemOperand mask][mem extras...][TA mask][ta extras...]
            GGML_ASSERT(wc >= 8);

            uint32_t mem_ಮುಖವಾಡ = code[pos + 6];
            size_t   cur      = pos + 7;
            // Each of these MemoryAccess bits (when set) carries one trailing operand.
            cur += (mem_ಮುಖವಾಡ & 0x2)     ? 1 : 0; // Aligned
            cur += (mem_ಮುಖವಾಡ & 0x8)     ? 1 : 0; // MakePointerAvailable
            cur += (mem_ಮುಖವಾಡ & 0x10)    ? 1 : 0; // MakePointerVisible
            cur += (mem_ಮುಖವಾಡ & 0x10000) ? 1 : 0; // AliasScopeINTELMask
            cur += (mem_ಮುಖವಾಡ & 0x20000) ? 1 : 0; // NoAliasINTELMask
            GGML_ASSERT(cur < pos + wc);

            uint32_t ta_ಮುಖವಾಡ = code[cur];
            ಆದರೆ ((ta_ಮುಖವಾಡ & k_Spv_Tensor_Addressing_Decode_ಸದಿಶ_Func_ಬಿಟ್) == 0) {
                pos += wc;
                ಮುಂದುವರಿ; // leave instruction inside the current unchanged run
            }

            flush_ಸತತ(pos);

            // Append unchanged prefix of the instruction (header through the mem-extras).
            size_t inst_ಆರಂಭ = ಹೊರಹೋಗು.size();
            size_t pre_n      = cur - pos;
            ಹೊರಹೋಗು.insert(ಹೊರಹೋಗು.end(), code + pos, code + pos + pre_n);

            // Emit TA mask with the DecodeVectorFunc bit cleared.
            ಹೊರಹೋಗು.push_back(ta_ಮುಖವಾಡ & ~k_Spv_Tensor_Addressing_Decode_ಸದಿಶ_Func_ಬಿಟ್);

            // TA extras: TensorView (0x1) and DecodeFunc (0x2) are kept verbatim;
            // DecodeVectorFunc (0x4) is dropped along with its trailing id operand.
            size_t keep_ta_extras = ((ta_ಮುಖವಾಡ & 0x1) ? 1 : 0) + ((ta_ಮುಖವಾಡ & 0x2) ? 1 : 0);
            ಆದರೆ (keep_ta_extras) {
                ಹೊರಹೋಗು.insert(ಹೊರಹೋಗು.end(), code + cur + 1, code + cur + 1 + keep_ta_extras);
            }

            GGML_ASSERT(wc == pre_n + 1 + keep_ta_extras + 1);

            // Patch the instruction header with the new (one-shorter) word count.
            uint32_t new_wc = wc - 1;
            ಹೊರಹೋಗು[inst_ಆರಂಭ] = (new_wc << spv::ಶಬ್ದ_ಎಣಿಕೆ_ಸರಿಸು) | op;

            pos += wc;
            ಸತತ_ಆರಂಭ = pos;
            ಮುಂದುವರಿ;
        }

        pos += wc;
    }

    flush_ಸತತ(ಶಬ್ದ_ಎಣಿಕೆ);
    ಹಿಂತಿರುಗಿ ನಿಜ;
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_vk_roll_bk_loop(ಸ್ಥಿರಾಂಕ uint32_t * code, size_t ಶಬ್ದ_ಎಣಿಕೆ, std::vector<uint32_t> & ಹೊರಹೋಗು) {
    ಆದರೆ (ಶಬ್ದ_ಎಣಿಕೆ < 5) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ರಚನೆ vk_spv_loop {
        size_t   ಶೀರ್ಷಿಕೆ;
        size_t   end;
        uint32_t control;
    };

    std::vector<vk_spv_loop> loops;

    // Collect a list of all loops in the module.
    ಗಾಗಿ (size_t pos = 5; pos < ಶಬ್ದ_ಎಣಿಕೆ; ) {
        ಸ್ಥಿರಾಂಕ uint32_t wc = code[pos] >> spv::ಶಬ್ದ_ಎಣಿಕೆ_ಸರಿಸು;
        ಸ್ಥಿರಾಂಕ uint32_t op = code[pos] & spv::Op_ಸಂಕೇತ_ಮುಖವಾಡ;
        ಆದರೆ (wc == 0 || pos + wc > ಶಬ್ದ_ಎಣಿಕೆ) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }

        ಆದರೆ (op == spv::Op_Loop_ವಿಲೀನ && wc >= 4) { loops.push_back({ pos, 0, code[pos + 3] }); }

        ಆದರೆ (op == spv::OpLabel && wc >= 2) {
            ಗಾಗಿ (ಸ್ವಯಂಚಾಲಿತ & l : loops) {
                ಆದರೆ (l.end == 0 && code[l.ಶೀರ್ಷಿಕೆ + 1] == code[pos + 1]) { l.end = pos; }
            }
        }

        pos += wc;
    }

    ಸ್ವಯಂಚಾಲಿತ encloses = [](ಸ್ಥಿರಾಂಕ vk_spv_loop & a, ಸ್ಥಿರಾಂಕ vk_spv_loop & b) {
        ಹಿಂತಿರುಗಿ a.ಶೀರ್ಷಿಕೆ < b.ಶೀರ್ಷಿಕೆ && b.ಶೀರ್ಷಿಕೆ < a.end;
    };

    // Find the BK loop.
    ಸ್ಥಿರಾಂಕ vk_spv_loop * bk = ಶೂನ್ಯಸೂಚಕ;
    ಗಾಗಿ (ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ & h : loops) {
        ಆದರೆ (h.control != spv::Loop_Control_Unroll_ಮುಖವಾಡ) {
            ಮುಂದುವರಿ;
        }
        ಸ್ಥಿರಾಂಕ vk_spv_loop * ಪೋಷಕ = ಶೂನ್ಯಸೂಚಕ;
        ತಾರ್ಕಿಕ has_ಮಗು = ಸುಳ್ಳು;
        ಗಾಗಿ (ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ & g : loops) {
            ಆದರೆ (encloses(g, h) && (!ಪೋಷಕ || g.ಶೀರ್ಷಿಕೆ > ಪೋಷಕ->ಶೀರ್ಷಿಕೆ)) {
                ಪೋಷಕ = &g;
            }
            ಆದರೆ (encloses(h, g)) {
                has_ಮಗು = ನಿಜ;
            }
        }
        // BK loop should be the last loop nested inside the loop with no hint
        // and have at least one child loop.
        ಆದರೆ (ಪೋಷಕ &&
            ಪೋಷಕ->control == spv::Loop_Control_ಮುಖವಾಡ_None &&
            has_ಮಗು &&
            (!bk || h.ಶೀರ್ಷಿಕೆ > bk->ಶೀರ್ಷಿಕೆ)) {
            bk = &h;
        }
    }
    ಆದರೆ (!bk) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    // set DontUnroll instead of Unroll
    ಹೊರಹೋಗು.ನಿಯೋಜಿಸು(code, code + ಶಬ್ದ_ಎಣಿಕೆ);
    ಹೊರಹೋಗು[bk->ಶೀರ್ಷಿಕೆ + 3] = spv::Loop_Control_Dont_Unroll_ಮುಖವಾಡ;
    ಹಿಂತಿರುಗಿ ನಿಜ;
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_vk_create_pipeline_func(vk_device& device, vk_pipeline& pipeline, size_t spv_ಗಾತ್ರ, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ* spv_ದತ್ತಾಂಶ, ಸ್ಥಿರಾಂಕ std::string entrypoint,
                                         uint32_t parameter_ಎಣಿಕೆ, std::array<uint32_t, 3> wg_denoms, std::vector<uint32_t> specialization_constants,
                                         ತಾರ್ಕಿಕ disable_robustness, ತಾರ್ಕಿಕ require_ತುಂಬಿದ_subgroups, uint32_t required_subgroup_ಗಾತ್ರ) {
    VK_LOG_DEBUG("ggml_vk_create_pipeline(" << device->name << ", " << pipeline->name << ", " << entrypoint << ", " << parameter_ಎಣಿಕೆ <<
                 ", (" << wg_denoms[0] << "," << wg_denoms[1] << "," << wg_denoms[2] << "), specialization_constants, " <<
                 disable_robustness << ", " << require_ತುಂಬಿದ_subgroups << ", " << required_subgroup_ಗಾತ್ರ << ")");
    GGML_ASSERT(parameter_ಎಣಿಕೆ > 0);
    GGML_ASSERT(parameter_ಎಣಿಕೆ <= MAX_PARAMETER_ಎಣಿಕೆ);
    GGML_ASSERT(wg_denoms[0] > 0 && wg_denoms[1] > 0 && wg_denoms[2] > 0); // NOLINT

    vk::ShaderModuleCreateInfo shader_module_create_info({}, spv_ಗಾತ್ರ, reinterpret_cast<ಸ್ಥಿರಾಂಕ uint32_t *>(spv_ದತ್ತಾಂಶ));

    // Patch SPIR-V to enable supported FP16 float controls, avoiding the need
    // for separate shader variants.
    std::vector<uint32_t> spirv;
    ಆದರೆ (device->float_controls_rte_fp16 || device->float_controls_denorm_preserve_fp16) {
        ಸ್ಥಿರಾಂಕ uint32_t* spv_ಪದಗಳು = reinterpret_cast<ಸ್ಥಿರಾಂಕ uint32_t *>(spv_ದತ್ತಾಂಶ);
        size_t ಶಬ್ದ_ಎಣಿಕೆ = spv_ಗಾತ್ರ / ಗಾತ್ರ(uint32_t);
        spirv.ನಿಯೋಜಿಸು(spv_ಪದಗಳು, spv_ಪದಗಳು + ಶಬ್ದ_ಎಣಿಕೆ);

        // Find insertion points respecting SPIR-V layout order:
        //   Header(5) -> OpCapability -> OpExtension -> ... -> OpEntryPoint -> OpExecutionMode -> ...
        size_t pos = 5; // skip header
        size_t cap_ಸೇರಿಸು_pos = pos;
        size_t ext_ಸೇರಿಸು_pos = pos;
        size_t exec_ಸೇರಿಸು_pos = pos;
        uint32_t ನಮೂದು_ಬಿಂದು_id = 0;

        ಆಗ (pos < spirv.size()) {
            uint32_t opcode = spirv[pos] & spv::Op_ಸಂಕೇತ_ಮುಖವಾಡ;
            uint32_t len    = spirv[pos] >> spv::ಶಬ್ದ_ಎಣಿಕೆ_ಸರಿಸು;
            ಆದರೆ (len == 0) ನಿಲ್ಲಿಸು;

            ಆದರೆ (opcode == spv::OpCapability) {
                cap_ಸೇರಿಸು_pos = pos + len;
                ext_ಸೇರಿಸು_pos = pos + len;
            } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (opcode == spv::OpExtension) {
                ext_ಸೇರಿಸು_pos = pos + len;
            } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (opcode == spv::Op_ನಮೂದು_ಬಿಂದು) {
                ನಮೂದು_ಬಿಂದು_id = spirv[pos + 2];
                exec_ಸೇರಿಸು_pos = pos + len;
            } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (opcode == spv::Op_Execution_ಬಹುಲಕ || opcode == spv::Op_Execution_ಬಹುಲಕ_Id) {
                exec_ಸೇರಿಸು_pos = pos + len;
            } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (ನಮೂದು_ಬಿಂದು_id != 0) {
                ನಿಲ್ಲಿಸು;
            }

            pos += len;
        }

        // Insert from latest position first so earlier indices stay valid.

        ಆದರೆ (device->float_controls_rte_fp16) {
            // OpExecutionMode %entrypoint RoundingModeRTE 16
            uint32_t exec_ಬಹುಲಕ[] = { (4u << spv::ಶಬ್ದ_ಎಣಿಕೆ_ಸರಿಸು) | spv::Op_Execution_ಬಹುಲಕ, ನಮೂದು_ಬಿಂದು_id, spv::Execution_ಬಹುಲಕ_Rounding_ಬಹುಲಕ_RTE, 16 };
            spirv.insert(spirv.begin() + exec_ಸೇರಿಸು_pos, std::begin(exec_ಬಹುಲಕ), std::end(exec_ಬಹುಲಕ));
        }

        ಆದರೆ (device->float_controls_denorm_preserve_fp16) {
            // OpExecutionMode %entrypoint DenormPreserve 16
            uint32_t exec_ಬಹುಲಕ[] = { (4u << spv::ಶಬ್ದ_ಎಣಿಕೆ_ಸರಿಸು) | spv::Op_Execution_ಬಹುಲಕ, ನಮೂದು_ಬಿಂದು_id, spv::Execution_ಬಹುಲಕ_Denorm_Preserve, 16 };
            spirv.insert(spirv.begin() + exec_ಸೇರಿಸು_pos, std::begin(exec_ಬಹುಲಕ), std::end(exec_ಬಹುಲಕ));
        }

        // OpExtension "SPV_KHR_float_controls"
        ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ ext_str[] = "SPV_KHR_float_controls";
        size_t ext_str_ಪದಗಳು = CEIL_DIV(ಗಾತ್ರ(ext_str), ಗಾತ್ರ(uint32_t));
        std::vector<uint32_t> extension(1 + ext_str_ಪದಗಳು, 0);
        extension[0] = (uint32_t)((1 + ext_str_ಪದಗಳು) << spv::ಶಬ್ದ_ಎಣಿಕೆ_ಸರಿಸು) | spv::OpExtension;
        ಸ್ಮರಣೆ_ನಕಲು(&extension[1], ext_str, ಗಾತ್ರ(ext_str));
        spirv.insert(spirv.begin() + ext_ಸೇರಿಸು_pos, extension.begin(), extension.end());

        ಆದರೆ (device->float_controls_rte_fp16) {
            // OpCapability RoundingModeRTE
            uint32_t capability[] = { (2u << spv::ಶಬ್ದ_ಎಣಿಕೆ_ಸರಿಸು) | spv::OpCapability, spv::Capability_Rounding_ಬಹುಲಕ_RTE };
            spirv.insert(spirv.begin() + cap_ಸೇರಿಸು_pos, std::begin(capability), std::end(capability));
        }

        ಆದರೆ (device->float_controls_denorm_preserve_fp16) {
            // OpCapability DenormPreserve
            uint32_t capability[] = { (2u << spv::ಶಬ್ದ_ಎಣಿಕೆ_ಸರಿಸು) | spv::OpCapability, spv::CapabilityDenormPreserve };
            spirv.insert(spirv.begin() + cap_ಸೇರಿಸು_pos, std::begin(capability), std::end(capability));
        }

        shader_module_create_info = vk::ShaderModuleCreateInfo({}, spirv.size() * ಗಾತ್ರ(uint32_t), spirv.ದತ್ತಾಂಶ());
    }

#ಆದರೆ defined(GGML_VULKAN_COOPMAT2_DECODE_ಸದಿಶ_GLSLC_SUPPORT)
    ಆದರೆ (device->coopmat2 && !device->coopmat2_decode_ಸದಿಶ) {
        ಸ್ಥಿರಾಂಕ uint32_t * src   = spirv.ಖಾಲಿಗಳು() ? reinterpret_cast<ಸ್ಥಿರಾಂಕ uint32_t *>(spv_ದತ್ತಾಂಶ) : spirv.ದತ್ತಾಂಶ();
        size_t           src_n = spirv.ಖಾಲಿಗಳು() ? spv_ಗಾತ್ರ / ಗಾತ್ರ(uint32_t) : spirv.size();
        std::vector<uint32_t> stripped;
        ಆದರೆ (ggml_vk_strip_decode_ಸದಿಶ(src, src_n, stripped)) {
            spirv = std::move(stripped);
            shader_module_create_info = vk::ShaderModuleCreateInfo({}, spirv.size() * ಗಾತ್ರ(uint32_t), spirv.ದತ್ತಾಂಶ());
        }
    }
#ಆದರೆ_ಅಂತ್ಯ

#ಆದರೆ VK_ಶೀರ್ಷಿಕೆ_ಆವೃತ್ತಿ >= 287
    // Roll the mul_mm BK loop on Asahi Linux. Skip bf16 and the mul_mmq pipelines.
    ಆದರೆ (device->driver_id == vk::DriverId::eMesaHoneykrisp &&
        pipeline->name.rfind("matmul", 0) == 0 &&
        pipeline->name.find("bf16") == std::string::npos &&
        pipeline->name.find("q8_1") == std::string::npos) {
        ಸ್ಥಿರಾಂಕ uint32_t * src   = spirv.ಖಾಲಿಗಳು() ? reinterpret_cast<ಸ್ಥಿರಾಂಕ uint32_t *>(spv_ದತ್ತಾಂಶ) : spirv.ದತ್ತಾಂಶ();
        size_t           src_n = spirv.ಖಾಲಿಗಳು() ? spv_ಗಾತ್ರ / ಗಾತ್ರ(uint32_t) : spirv.size();
        std::vector<uint32_t> rolled;
        ಆದರೆ (ggml_vk_roll_bk_loop(src, src_n, rolled)) {
            spirv = std::move(rolled);
            shader_module_create_info = vk::ShaderModuleCreateInfo({}, spirv.size() * ಗಾತ್ರ(uint32_t), spirv.ದತ್ತಾಂಶ());
        }
    }
#ಆದರೆ_ಅಂತ್ಯ

    ಪ್ರಯತ್ನಿಸು {
        pipeline->shader_module = device->device.createShaderModule(shader_module_create_info);
    } ಹಿಡಿ (ಸ್ಥಿರಾಂಕ vk::SystemError& e) {
        std::cerr << "ggml_vulkan: shader module creation failed for " << pipeline->name << ": " << e.what() << std::ಸಾಲು_ಅಂತ್ಯ;
        ಎಸೆ;
    }

    vk::ತಳ್ಳು_Constant_Range pcr(
        vk::Shader_Stage_Flag_ಬಿಟ್ಟುಗಳು::eCompute,
        0,
        pipeline->ತಳ್ಳು_constant_ಗಾತ್ರ
    );

    vk::PipelineLayoutCreateInfo pipeline_layout_create_info(vk::PipelineLayoutCreateFlags(), device->dsl, pcr);
    ಪ್ರಯತ್ನಿಸು {
        pipeline->layout = device->device.createPipelineLayout(pipeline_layout_create_info);
    } ಹಿಡಿ (ಸ್ಥಿರಾಂಕ vk::SystemError& e) {
        std::cerr << "ggml_vulkan: pipeline layout creation failed for " << pipeline->name << ": " << e.what() << std::ಸಾಲು_ಅಂತ್ಯ;
        ಎಸೆ;
    }

    std::vector<vk::Specialization_Map_ನಮೂದು> specialization_entries(specialization_constants.size());

    ಗಾಗಿ (size_t i = 0; i < specialization_constants.size(); i++) {
        specialization_entries[i].constantID = i;
        specialization_entries[i].ಸ್ಥಾನಾಂತರ = i * ಗಾತ್ರ(uint32_t);
        specialization_entries[i].size = ಗಾತ್ರ(uint32_t);
    }

    vk::SpecializationInfo specialization_info(
        specialization_entries.size(),
        specialization_entries.ದತ್ತಾಂಶ(),
        specialization_constants.size() * ಗಾತ್ರ(uint32_t),
        specialization_constants.ದತ್ತಾಂಶ()
    );

    vk::PipelineShaderStageCreateFlags pipeline_shader_stage_create_flags{};

    ಆದರೆ (device->subgroup_require_ತುಂಬಿದ_support && require_ತುಂಬಿದ_subgroups) {
        pipeline_shader_stage_create_flags |= vk::Pipeline_Shader_Stage_Create_Flag_ಬಿಟ್ಟುಗಳು::e_Require_ತುಂಬಿದ_Subgroups_EXT;
    }

    vk::PipelineShaderStageCreateInfo pipeline_shader_create_info(
            pipeline_shader_stage_create_flags,
            vk::Shader_Stage_Flag_ಬಿಟ್ಟುಗಳು::eCompute,
            pipeline->shader_module,
            entrypoint.c_str(),
            &specialization_info);

    vk::Pipeline_Shader_Stage_Required_Subgroup_ಗಾತ್ರ_Create_Info_EXT pipeline_shader_stage_required_subgroup_ಗಾತ್ರ_create_info;
    pipeline_shader_stage_required_subgroup_ಗಾತ್ರ_create_info.required_Subgroup_ಗಾತ್ರ = required_subgroup_ಗಾತ್ರ;
    ಆದರೆ (device->subgroup_ಗಾತ್ರ_control && required_subgroup_ಗಾತ್ರ > 0) {
        GGML_ASSERT(device->subgroup_min_ಗಾತ್ರ <= required_subgroup_ಗಾತ್ರ && required_subgroup_ಗಾತ್ರ <= device->subgroup_max_ಗಾತ್ರ);
        pipeline_shader_create_info.ಗಣ_PNext(&pipeline_shader_stage_required_subgroup_ಗಾತ್ರ_create_info);
    }

    vk::ComputePipelineCreateInfo compute_pipeline_create_info(
        device->pipeline_executable_properties_support ?
            vk::Pipeline_Create_Flag_ಬಿಟ್ಟುಗಳು::eCaptureStatisticsKHR :
            vk::PipelineCreateFlags{},
        pipeline_shader_create_info,
        pipeline->layout);

    vk::PipelineRobustnessCreateInfoEXT rci;

    ಆದರೆ (device->pipeline_robustness && disable_robustness) {
        rci.storageBuffers = vk::Pipeline_Robustness_ಬಫರ್_Behavior_EXT::eDisabled;
        rci.ಏಕರೂಪ_Buffers = vk::Pipeline_Robustness_ಬಫರ್_Behavior_EXT::eDisabled;
        compute_pipeline_create_info.ಗಣ_PNext(&rci);
    }

#ಆದರೆ defined(VK_EXT_shader_64bit_indexing)
    vk::PipelineCreateFlags2CreateInfo pipelineFlags2CreateInfo;
    ಆದರೆ (pipeline->is_64b_indexing)
    {
        pipelineFlags2CreateInfo.flags = vk::PipelineCreateFlagBits2::e64_ಬಿಟ್_Indexing_EXT;
        ಆದರೆ (device->pipeline_executable_properties_support) {
            pipelineFlags2CreateInfo.flags |= vk::PipelineCreateFlagBits2::eCaptureStatisticsKHR;
        }
        pipelineFlags2CreateInfo.ಗಣ_PNext(compute_pipeline_create_info.p_ಮುಂದಿನ);
        compute_pipeline_create_info.ಗಣ_PNext(&pipelineFlags2CreateInfo);
    }
#ಆದರೆ_ಅಂತ್ಯ

    ಪ್ರಯತ್ನಿಸು {
        pipeline->pipeline = device->device.createComputePipeline(VK_NULL_HANDLE, compute_pipeline_create_info).value;
    } ಹಿಡಿ (ಸ್ಥಿರಾಂಕ vk::SystemError& e) {
        std::cerr << "ggml_vulkan: compute pipeline creation failed for " << pipeline->name << ": " << e.what() << std::ಸಾಲು_ಅಂತ್ಯ;
        ಎಸೆ;
    }

    ಆದರೆ (vk_instance.debug_utils_support) {
        vk::DebugUtilsObjectNameInfoEXT duoni;
        duoni.objectType = vk::ObjectType::ePipeline;
        duoni.pObjectName = pipeline->name.c_str();
        duoni.objectHandle = /*reinterpret_cast*/(uint64_t)(static_cast<VkPipeline>(pipeline->pipeline));
        vk_instance.pfn_vk_ಗಣ_Debug_Utils_Object_Name_EXT(device->device, &static_cast<VkDebugUtilsObjectNameInfoEXT &>(duoni));
    }

    ಆದರೆ (device->pipeline_executable_properties_support) {
        vk::PipelineExecutableInfoKHR executableInfo;
        executableInfo.pipeline = pipeline->pipeline;

        ಸ್ವಯಂಚಾಲಿತ statistics = device->device.getPipelineExecutableStatisticsKHR(executableInfo);

        ತಾರ್ಕಿಕ print_stats = !vk_pipeline_stats_filter.ಖಾಲಿಗಳು() &&
                           pipeline->name.find(vk_pipeline_stats_filter) != std::string::npos;
        ಆದರೆ (print_stats) {
            std::cerr << "ggml_vulkan: pipeline stats for " << pipeline->name << ":" << std::ಸಾಲು_ಅಂತ್ಯ;
        }

        ಗಾಗಿ (ಸ್ವಯಂಚಾಲಿತ & s : statistics) {
            ಆದರೆ (print_stats) {
                std::cerr << "ggml_vulkan:   " << s.name.ದತ್ತಾಂಶ() << ": ";
                ಆಯ್ಕೆ (s.format) {
                    ಸ್ಥಿತಿ vk::PipelineExecutableStatisticFormatKHR::eBool32:
                        std::cerr << (s.value.b32 ? "true" : "false");
                        ನಿಲ್ಲಿಸು;
                    ಸ್ಥಿತಿ vk::PipelineExecutableStatisticFormatKHR::eInt64:
                        std::cerr << s.value.i64;
                        ನಿಲ್ಲಿಸು;
                    ಸ್ಥಿತಿ vk::PipelineExecutableStatisticFormatKHR::eUint64:
                        std::cerr << s.value.u64;
                        ನಿಲ್ಲಿಸು;
                    ಸ್ಥಿತಿ vk::PipelineExecutableStatisticFormatKHR::eFloat64:
                        std::cerr << s.value.f64;
                        ನಿಲ್ಲಿಸು;
                }
                std::cerr << std::ಸಾಲು_ಅಂತ್ಯ;
            }
            // "Register Count" is reported by NVIDIA drivers.
            ಆದರೆ (strcmp(s.name, "Register Count") == 0) {
                VK_LOG_DEBUG(pipeline->name << " " << s.name << ": " << s.value.u64 << " registers");
                pipeline->ರಿಜಿಸ್ಟರ್_ಎಣಿಕೆ = (uint32_t)s.value.u64;
            }
        }
    }

    {
        std::lock_guard<std::mutex> guard(device->compile_mutex);
        device->all_pipelines.push_back(pipeline);
        pipeline->compiled = ನಿಜ;
        pipeline->compile_pending = ಸುಳ್ಳು;
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
    ctx->pipeline_descriptor_ಗಣ_requirements += n;
    ಆದರೆ (!pipeline->compiled) {
        ggml_vk_ತುಂಬು_shaders(ctx->device, pipeline);
    }
    ggml_pipeline_allocate_descriptor_sets(ctx);
}

ಶೂನ್ಯ ggml_pipeline_allocate_descriptor_sets(ggml_backend_vk_context * ctx) {

    ಆದರೆ (ctx->descriptor_sets.size() >= ctx->pipeline_descriptor_ಗಣ_requirements) {
        // Enough descriptors are available
        ಹಿಂತಿರುಗಿ;
    }

    vk_device& device = ctx->device;

    // Grow by 50% to avoid frequent allocations
    uint32_t needed = std::max(3 * ctx->descriptor_sets.size() / 2, size_t{ctx->pipeline_descriptor_ಗಣ_requirements});
    uint32_t ಗಮ್ಯ_alloc = needed - ctx->descriptor_sets.size();
    uint32_t pool_ಉಳಿದ = VK_DEVICE_DESCRIPTOR_POOL_ಗಾತ್ರ - ctx->descriptor_sets.size() % VK_DEVICE_DESCRIPTOR_POOL_ಗಾತ್ರ;
    uint32_t pool_idx = ctx->descriptor_sets.size() / VK_DEVICE_DESCRIPTOR_POOL_ಗಾತ್ರ;

    ಆಗ (ಗಮ್ಯ_alloc > 0) {
        ಸ್ಥಿರಾಂಕ uint32_t alloc_ಎಣಿಕೆ = std::min(pool_ಉಳಿದ, ಗಮ್ಯ_alloc);
        ಗಮ್ಯ_alloc -= alloc_ಎಣಿಕೆ;
        pool_ಉಳಿದ = VK_DEVICE_DESCRIPTOR_POOL_ಗಾತ್ರ;

        ಆದರೆ (pool_idx >= ctx->descriptor_pools.size()) {
            vk::Descriptor_Pool_ಗಾತ್ರ descriptor_pool_ಗಾತ್ರ(vk::DescriptorType::e_Storage_ಬಫರ್, MAX_PARAMETER_ಎಣಿಕೆ * VK_DEVICE_DESCRIPTOR_POOL_ಗಾತ್ರ);
            vk::DescriptorPoolCreateInfo descriptor_pool_create_info({}, VK_DEVICE_DESCRIPTOR_POOL_ಗಾತ್ರ, descriptor_pool_ಗಾತ್ರ);
            ctx->descriptor_pools.push_back(device->device.createDescriptorPool(descriptor_pool_create_info));
        }

        std::vector<vk::Descriptor_ಗಣ_Layout> layouts(alloc_ಎಣಿಕೆ);
        ಗಾಗಿ (uint32_t i = 0; i < alloc_ಎಣಿಕೆ; i++) {
            layouts[i] = device->dsl;
        }
        vk::Descriptor_ಗಣ_Allocate_Info descriptor_ಗಣ_alloc_info(ctx->descriptor_pools[pool_idx], alloc_ಎಣಿಕೆ, layouts.ದತ್ತಾಂಶ());
        std::vector<vk::Descriptor_ಗಣ> sets = device->device.allocateDescriptorSets(descriptor_ಗಣ_alloc_info);
        ctx->descriptor_sets.insert(ctx->descriptor_sets.end(), sets.begin(), sets.end());

        pool_idx++;
    }
    ctx->descriptor_ಗಣ_bindings.resize(ctx->descriptor_sets.size());
}

ಸ್ಥಿರ vk_command_ಬಫರ್* ggml_vk_create_cmd_ಬಫರ್(vk_device& device, vk_command_pool& p) {
    VK_LOG_DEBUG("ggml_vk_create_cmd_buffer()");
    vk::Command_ಬಫರ್_Allocate_Info command_ಬಫರ್_alloc_info(
        p.pool,
        vk::Command_ಬಫರ್_Level::ePrimary,
        1);
    ಸ್ಥಿರಾಂಕ std::vector<vk::Command_ಬಫರ್> cmd_buffers = device->device.allocateCommandBuffers(command_ಬಫರ್_alloc_info);
    p.cmd_buffers.push_back({ cmd_buffers.front(), 0, ನಿಜ });
    ಹಿಂತಿರುಗಿ &p.cmd_buffers[p.cmd_buffers.size()-1];
}

ಶೂನ್ಯ ggml_vk_submit(vk_context& ctx, vk::Fence fence) {
    ಆದರೆ (ctx->seqs.ಖಾಲಿಗಳು()) {
        ಆದರೆ (fence) {
            ctx->p->q->handle->submit({}, fence);
        }
        ಹಿಂತಿರುಗಿ;
    }
    VK_LOG_DEBUG("ggml_vk_submit(" << ctx << ", " << fence << ")");

    std::vector<std::vector<uint64_t>> tl_wait_vals;
    std::vector<std::vector<uint64_t>> tl_signal_vals;
    std::vector<std::vector<vk::Semaphore>> tl_wait_semaphores;
    std::vector<std::vector<vk::Semaphore>> tl_signal_semaphores;
    std::vector<vk::TimelineSemaphoreSubmitInfo> tl_submit_infos;
    std::vector<vk::SubmitInfo> submit_infos;
    ಪೂರ್ಣಾಂಕ idx = -1;
    std::vector<std::vector<vk::PipelineStageFlags>> stage_flags;

    size_t reserve = 0;

    ಗಾಗಿ (ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ& sequence : ctx->seqs) {
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

    ಗಾಗಿ (ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ& sequence : ctx->seqs) {
        ಗಾಗಿ (ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ& submission : sequence) {
            stage_flags.push_back({});
            idx++;
            tl_wait_vals.push_back({});
            tl_wait_semaphores.push_back({});
            tl_signal_vals.push_back({});
            tl_signal_semaphores.push_back({});
            ಗಾಗಿ (size_t i = 0; i < submission.wait_semaphores.size(); i++) {
                stage_flags[idx].push_back(ctx->p->q->stage_flags);
                tl_wait_vals[idx].push_back(submission.wait_semaphores[i].value);
                tl_wait_semaphores[idx].push_back(submission.wait_semaphores[i].s);
            }
            ಗಾಗಿ (size_t i = 0; i < submission.signal_semaphores.size(); i++) {
                tl_signal_vals[idx].push_back(submission.signal_semaphores[i].value);
                tl_signal_semaphores[idx].push_back(submission.signal_semaphores[i].s);
            }
            tl_submit_infos.push_back({
                (uint32_t) submission.wait_semaphores.size(),
                tl_wait_vals[idx].ದತ್ತಾಂಶ(),
                (uint32_t) submission.signal_semaphores.size(),
                tl_signal_vals[idx].ದತ್ತಾಂಶ(),
            });
            tl_submit_infos[idx].sType = vk::StructureType::eTimelineSemaphoreSubmitInfo;
            tl_submit_infos[idx].p_ಮುಂದಿನ = ಶೂನ್ಯಸೂಚಕ;
            vk::SubmitInfo si{
                (uint32_t) submission.wait_semaphores.size(),
                tl_wait_semaphores[idx].ದತ್ತಾಂಶ(),
                stage_flags[idx].ದತ್ತಾಂಶ(),
                1,
                &submission.ಬಫರ್->buf,
                (uint32_t) submission.signal_semaphores.size(),
                tl_signal_semaphores[idx].ದತ್ತಾಂಶ(),
            };
            si.ಗಣ_PNext(&tl_submit_infos[idx]);
            submit_infos.push_back(si);
        }
    }

    ctx->p->q->handle->submit(submit_infos, fence);

    ctx->seqs.clear();
}

uint32_t ggml_vk_ಕಂಡುಹಿಡಿ_ಸರತಿ_family_ಸೂಚ್ಯಂಕ(std::vector<vk::ಸರತಿ_Family_Properties>& ಸರತಿ_family_props, ಸ್ಥಿರಾಂಕ vk::ಸರತಿ_Flags& required, ಸ್ಥಿರಾಂಕ vk::ಸರತಿ_Flags& avoid, int32_t compute_ಸೂಚ್ಯಂಕ, uint32_t min_ಅಂಕೆ_queues) {
    VK_LOG_DEBUG("ggml_vk_find_queue_family_index()");
    ಸ್ಥಿರಾಂಕ uint32_t qfsize = ಸರತಿ_family_props.size();

    // Try with avoid preferences first
    ಗಾಗಿ (uint32_t i = 0; i < qfsize; i++) {
        ಆದರೆ (ಸರತಿ_family_props[i].ಸರತಿ_ಎಣಿಕೆ >= min_ಅಂಕೆ_queues && (compute_ಸೂಚ್ಯಂಕ < 0 || i != (uint32_t) compute_ಸೂಚ್ಯಂಕ) && ಸರತಿ_family_props[i].ಸರತಿ_Flags_2 & required && !(ಸರತಿ_family_props[i].ಸರತಿ_Flags_2 & avoid)) {
            ಹಿಂತಿರುಗಿ i;
        }
    }

    // Fall back to only required
    ಗಾಗಿ (size_t i = 0; i < qfsize; i++) {
        ಆದರೆ (ಸರತಿ_family_props[i].ಸರತಿ_ಎಣಿಕೆ >= min_ಅಂಕೆ_queues && (compute_ಸೂಚ್ಯಂಕ < 0 || i != (uint32_t) compute_ಸೂಚ್ಯಂಕ) && ಸರತಿ_family_props[i].ಸರತಿ_Flags_2 & required) {
            ಹಿಂತಿರುಗಿ i;
        }
    }

    // Fall back to reusing compute queue
    ಗಾಗಿ (size_t i = 0; i < qfsize; i++) {
        ಆದರೆ (ಸರತಿ_family_props[i].ಸರತಿ_ಎಣಿಕೆ >= min_ಅಂಕೆ_queues && ಸರತಿ_family_props[i].ಸರತಿ_Flags_2 & required) {
            ಹಿಂತಿರುಗಿ i;
        }
    }

    // Fall back to ignoring min_num_queries
    ಗಾಗಿ (size_t i = 0; i < qfsize; i++) {
        ಆದರೆ (ಸರತಿ_family_props[i].ಸರತಿ_Flags_2 & required) {
            ಹಿಂತಿರುಗಿ i;
        }
    }

    // All commands that are allowed on a queue that supports transfer operations are also allowed on a queue that supports either graphics or compute operations.
    // Thus, if the capabilities of a queue family include VK_QUEUE_GRAPHICS_BIT or VK_QUEUE_COMPUTE_BIT, then reporting the VK_QUEUE_TRANSFER_BIT capability separately for that queue family is optional.
    ಆದರೆ (compute_ಸೂಚ್ಯಂಕ >= 0) {
        ಹಿಂತಿರುಗಿ compute_ಸೂಚ್ಯಂಕ;
    }

    std::cerr << "ggml_vulkan: No suitable queue family index found." << std::ಸಾಲು_ಅಂತ್ಯ;

    ಗಾಗಿ(ಸ್ವಯಂಚಾಲಿತ &q_family : ಸರತಿ_family_props) {
        std::cerr << "Queue number: "  + std::to_string(q_family.ಸರತಿ_ಎಣಿಕೆ) << " flags: " + to_string(q_family.ಸರತಿ_Flags_2) << std::ಸಾಲು_ಅಂತ್ಯ;
    }
    abort();
}

std::unique_ptr<vk_ಸರತಿ> ggml_vk_create_ಸರತಿ(vk_device& device, uint32_t ಸರತಿ_family_ಸೂಚ್ಯಂಕ, uint32_t ಸರತಿ_ಸೂಚ್ಯಂಕ, vk::PipelineStageFlags&& stage_flags, ತಾರ್ಕಿಕ transfer_only) {
    VK_LOG_DEBUG("ggml_vk_create_queue()");
    std::lock_guard<std::recursive_mutex> guard(device->mutex);

    ಸ್ವಯಂಚಾಲಿತ q = std::make_unique<vk_ಸರತಿ>();
    q->ಸರತಿ_family_ಸೂಚ್ಯಂಕ = ಸರತಿ_family_ಸೂಚ್ಯಂಕ;
    q->transfer_only = transfer_only;

    std::shared_ptr<vk_ಸರತಿ_handle> h;
    vk::Device_ಸರತಿ_Info2 ಸರತಿ_info2{};
    ಸರತಿ_info2.ಸರತಿ_Family_ಸೂಚ್ಯಂಕ = ಸರತಿ_family_ಸೂಚ್ಯಂಕ;
    ಸರತಿ_info2.ಸರತಿ_ಸೂಚ್ಯಂಕ_2 = ಸರತಿ_ಸೂಚ್ಯಂಕ;

    ಆದರೆ (device->has_internally_synchronized_queues) {
        h = std::make_shared<vk_ಸರತಿ_handle_unsynchronized>();
        ಸರತಿ_info2.flags = eInternallySynchronizedKHR;
    } ಇಲ್ಲದಿದ್ದರೆ {
        h = std::make_shared<vk_ಸರತಿ_handle_synchronized>();
    }

    h->queue = device->device.getQueue2(ಸರತಿ_info2);
    h->device = device;
    // Avoid concurrent submissions on NVIDIA due to driver bug.
    ಆದರೆ (device->vendor_id == VK_VENDOR_ID_NVIDIA) {
        h->device_submit_mutex = &device->ಸರತಿ_submit_mutex;
    }
    q->handle = h;

    q->cmd_pool.init(device, q.get());

    q->stage_flags = stage_flags;
    ಹಿಂತಿರುಗಿ q;
}

std::unique_ptr<vk_ಸರತಿ> ggml_vk_create_aliased_ಸರತಿ(vk_device& device, ಸ್ಥಿರಾಂಕ std::unique_ptr<vk_ಸರತಿ>& ಮೂಲ) {
    std::lock_guard<std::recursive_mutex> guard(device->mutex);
    ಸ್ವಯಂಚಾಲಿತ q = std::make_unique<vk_ಸರತಿ>();
    q->handle = ಮೂಲ->handle;
    q->ಸರತಿ_family_ಸೂಚ್ಯಂಕ = ಮೂಲ->ಸರತಿ_family_ಸೂಚ್ಯಂಕ;
    q->stage_flags = ಮೂಲ->stage_flags;
    q->transfer_only = ಮೂಲ->transfer_only;
    q->cmd_pool.init(device, q.get());
    ಹಿಂತಿರುಗಿ q;
}

vk_context ggml_vk_create_context(ggml_backend_vk_context * ctx, vk_command_pool& p) {
    vk_context ಫಲಿತಾಂಶ = std::make_shared<vk_context_struct>();
    VK_LOG_DEBUG("ggml_vk_create_context(" << ಫಲಿತಾಂಶ << ")");
    ctx->gc.contexts.emplace_back(ಫಲಿತಾಂಶ);
    ಫಲಿತಾಂಶ->p = &p;
    ಹಿಂತಿರುಗಿ ಫಲಿತಾಂಶ;
}

vk_context ggml_vk_create_temporary_context(vk_command_pool& p) {
    vk_context ಫಲಿತಾಂಶ = std::make_shared<vk_context_struct>();
    VK_LOG_DEBUG("ggml_vk_create_temporary_context(" << ಫಲಿತಾಂಶ << ")");
    ಫಲಿತಾಂಶ->p = &p;
    ಹಿಂತಿರುಗಿ ಫಲಿತಾಂಶ;
}

ಸ್ಥಿರ vk_semaphore * ggml_vk_create_ದ್ವಿಮಾನ_semaphore(ggml_backend_vk_context * ctx) {
    VK_LOG_DEBUG("ggml_vk_create_timeline_semaphore()");
    vk::SemaphoreTypeCreateInfo tci{ vk::SemaphoreType::e_ದ್ವಿಮಾನ, 0 };
    vk::SemaphoreCreateInfo ci{};
    ci.ಗಣ_PNext(&tci);
    vk::Semaphore semaphore = ctx->device->device.createSemaphore(ci);
    ctx->gc.semaphores.push_back({ semaphore, 0 });
    ಹಿಂತಿರುಗಿ &ctx->gc.semaphores[ctx->gc.semaphores.size() - 1];
}

ಸ್ಥಿರ vk_semaphore * ggml_vk_create_timeline_semaphore(ggml_backend_vk_context * ctx) {
    VK_LOG_DEBUG("ggml_vk_create_timeline_semaphore()");
    ಆದರೆ (ctx->semaphore_idx >= ctx->gc.tl_semaphores.size()) {
        vk::SemaphoreTypeCreateInfo tci{ vk::SemaphoreType::eTimeline, 0 };
        vk::SemaphoreCreateInfo ci{};
        ci.ಗಣ_PNext(&tci);
        vk::Semaphore semaphore = ctx->device->device.createSemaphore(ci);
        ctx->gc.tl_semaphores.push_back({ semaphore, 0 });
    }
    ಹಿಂತಿರುಗಿ &ctx->gc.tl_semaphores[ctx->semaphore_idx++];
}

ಸ್ಥಿರ vk::Event ggml_vk_create_event(ggml_backend_vk_context * ctx) {
    ಆದರೆ (ctx->event_idx >= ctx->gc.events.size()) {
        ctx->gc.events.push_back(ctx->device->device.createEvent({}));
    }
    ಹಿಂತಿರುಗಿ ctx->gc.events[ctx->event_idx++];
}

ಶೂನ್ಯ ggml_vk_command_pool_cleanup(vk_device& device, vk_command_pool& p) {
    VK_LOG_DEBUG("ggml_vk_command_pool_cleanup()");

    // Requires command buffers to be done
    device->device.resetCommandPool(p.pool);
    // Don't clear the command buffers and mark them as not in use.
    // This allows us to reuse them
    ಗಾಗಿ (ಸ್ವಯಂಚಾಲಿತ& cmd_ಬಫರ್ : p.cmd_buffers) {
        cmd_ಬಫರ್.ಒಳ_use = ಸುಳ್ಳು;
    }
}

ಶೂನ್ಯ ggml_vk_ಸರತಿ_command_pools_cleanup(vk_device& device) {
    VK_LOG_DEBUG("ggml_vk_queue_command_pools_cleanup()");

    // Arbitrary frequency to cleanup/reuse command buffers
    ಸ್ಥಿರ constexpr uint32_t cleanup_frequency = 10;

    ಆದರೆ (device->compute_ಸರತಿ && device->compute_ಸರತಿ->cmd_pool.buffers_ಒಳ_use() >= cleanup_frequency) {
        ggml_vk_command_pool_cleanup(device, device->compute_ಸರತಿ->cmd_pool);
    }
    ಆದರೆ (device->transfer_ಸರತಿ && device->transfer_ಸರತಿ->cmd_pool.buffers_ಒಳ_use() >= cleanup_frequency) {
        ggml_vk_command_pool_cleanup(device, device->transfer_ಸರತಿ->cmd_pool);
    }
}

vk_subbuffer ggml_vk_subbuffer(ಸ್ಥಿರಾಂಕ ggml_backend_vk_context* ctx, ಸ್ಥಿರಾಂಕ vk_ಬಫರ್& buf, size_t ಸ್ಥಾನಾಂತರ) {
    ಹಿಂತಿರುಗಿ { buf, ಸ್ಥಾನಾಂತರ, ggml_vk_get_max_ಬಫರ್_range(ctx, buf, ಸ್ಥಾನಾಂತರ) };
}

ಶೂನ್ಯ ggml_vk_sync_buffers(ggml_backend_vk_context* ctx, vk_context& subctx) {
    VK_LOG_DEBUG("ggml_vk_sync_buffers()");

    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ transfer_ಸರತಿ = subctx->p->q->transfer_only;

    ಆದರೆ (ctx) {
        ctx->prealloc_x_need_sync = ctx->prealloc_y_need_sync = ctx->prealloc_ವಿಭಜಿಸು_k_need_sync = ಸುಳ್ಳು;
    }

    subctx->s->ಬಫರ್->buf.pipelineBarrier(
        subctx->p->q->stage_flags,
        subctx->p->q->stage_flags,
        {},
        { {
          { !transfer_ಸರತಿ ? (vk::Access_Flag_ಬಿಟ್ಟುಗಳು::eShaderRead | vk::Access_Flag_ಬಿಟ್ಟುಗಳು::eShaderWrite | vk::Access_Flag_ಬಿಟ್ಟುಗಳು::eTransferRead | vk::Access_Flag_ಬಿಟ್ಟುಗಳು::eTransferWrite) : (vk::Access_Flag_ಬಿಟ್ಟುಗಳು::eTransferRead | vk::Access_Flag_ಬಿಟ್ಟುಗಳು::eTransferWrite) },
          { !transfer_ಸರತಿ ? (vk::Access_Flag_ಬಿಟ್ಟುಗಳು::eShaderRead | vk::Access_Flag_ಬಿಟ್ಟುಗಳು::eShaderWrite | vk::Access_Flag_ಬಿಟ್ಟುಗಳು::eTransferRead | vk::Access_Flag_ಬಿಟ್ಟುಗಳು::eTransferWrite) : (vk::Access_Flag_ಬಿಟ್ಟುಗಳು::eTransferRead | vk::Access_Flag_ಬಿಟ್ಟುಗಳು::eTransferWrite) }
        } },
        {},
        {}
    );
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_vk_reset_event(vk_context& ctx, vk::Event& event) {
    VK_LOG_DEBUG("ggml_vk_set_event()");

    ctx->s->ಬಫರ್->buf.resetEvent(
        event,
        ctx->p->q->stage_flags
    );
}

ಶೂನ್ಯ ggml_vk_ಗಣ_event(vk_context& ctx, vk::Event& event) {
    VK_LOG_DEBUG("ggml_vk_set_event()");

    ctx->s->ಬಫರ್->buf.ಗಣ_Event(
        event,
        ctx->p->q->stage_flags
    );
}

ಶೂನ್ಯ ggml_vk_wait_events(vk_context& ctx, std::vector<vk::Event>&& events) {
    VK_LOG_DEBUG("ggml_vk_wait_events()");
    ಆದರೆ (events.ಖಾಲಿಗಳು()) {
        ಹಿಂತಿರುಗಿ;
    }

    ctx->s->ಬಫರ್->buf.waitEvents(
        events,
        ctx->p->q->stage_flags,
        ctx->p->q->stage_flags,
        {},
        {},
        {}
    );
}

ಸ್ಥಿರ vk_fa_tuning_params get_fa_tuning_params_scalar(ಸ್ಥಿರಾಂಕ vk_device& device, uint32_t hsk, uint32_t hsv, uint32_t n_ಅಡ್ಡಸಾಲುಗಳು, uint32_t n_kv, ggml_type k_type, ggml_type v_type, ತಾರ್ಕಿಕ f32acc) {

    vk_fa_tuning_params ಫಲಿತಾಂಶ{};
    ಫಲಿತಾಂಶ.path = FA_SCALAR;

    ಆದರೆ (device->vendor_id == VK_VENDOR_ID_INTEL) {
        // Disable subgroup use due to performance issues when enforcing subgroup sizes
        ಫಲಿತಾಂಶ.subgroup_ಗಾತ್ರ = 32;
        ಫಲಿತಾಂಶ.disable_subgroups = ನಿಜ;
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (device->vendor_id == VK_VENDOR_ID_AMD && device->architecture != AMD_GCN) {
        ಫಲಿತಾಂಶ.subgroup_ಗಾತ್ರ = n_ಅಡ್ಡಸಾಲುಗಳು < 4 ? 32 : device->subgroup_ಗಾತ್ರ;
    } ಇಲ್ಲದಿದ್ದರೆ {
        ಫಲಿತಾಂಶ.subgroup_ಗಾತ್ರ = device->subgroup_ಗಾತ್ರ;
    }

    // Row split splits the workgroup so that synchronization only has to happen within subgroups, which avoids barriers
    uint32_t ಅಡ್ಡಸಾಲು_ವಿಭಜಿಸು_max_hsk = 64;
    ಆದರೆ (device->vendor_id == VK_VENDOR_ID_AMD && device->architecture != AMD_GCN && !device->uma) {
        ಅಡ್ಡಸಾಲು_ವಿಭಜಿಸು_max_hsk = n_ಅಡ್ಡಸಾಲುಗಳು <= 8 ? 64 : 128;
    }
    ಫಲಿತಾಂಶ.ಅಡ್ಡಸಾಲು_ವಿಭಜಿಸು = (n_ಅಡ್ಡಸಾಲುಗಳು < 4 || hsk <= ಅಡ್ಡಸಾಲು_ವಿಭಜಿಸು_max_hsk) ? 1 : 4;

    ಆದರೆ (ಫಲಿತಾಂಶ.subgroup_ಗಾತ್ರ > 32 && (n_ಅಡ್ಡಸಾಲುಗಳು < 4 || hsk < (ಫಲಿತಾಂಶ.ಅಡ್ಡಸಾಲು_ವಿಭಜಿಸು == 1 ? 128 : 64))) {
        ಫಲಿತಾಂಶ.workgroup_ಗಾತ್ರ = ಫಲಿತಾಂಶ.subgroup_ಗಾತ್ರ * 2;
    } ಇಲ್ಲದಿದ್ದರೆ {
        ಫಲಿತಾಂಶ.workgroup_ಗಾತ್ರ = ಫಲಿತಾಂಶ.subgroup_ಗಾತ್ರ * 4;
    }

    ಸ್ಥಿರಾಂಕ uint32_t D = hsk | hsv;

    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ reduce_ಖಂಡ_ಅಡ್ಡಸಾಲುಗಳು = D & 8 || n_kv < 1024 || device->vendor_id == VK_VENDOR_ID_INTEL;

    ಆದರೆ (n_ಅಡ್ಡಸಾಲುಗಳು == 1) {
        ಫಲಿತಾಂಶ.ಖಂಡ_ಅಡ್ಡಸಾಲುಗಳು = 1;
        ಫಲಿತಾಂಶ.ಖಂಡ_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ = 64;
    } ಇಲ್ಲದಿದ್ದರೆ {
        // row_split 1 means higher register use per row, so block size has to be adjusted
        ಆದರೆ (ಫಲಿತಾಂಶ.ಅಡ್ಡಸಾಲು_ವಿಭಜಿಸು == 1) {
            ಫಲಿತಾಂಶ.ಖಂಡ_ಅಡ್ಡಸಾಲುಗಳು = n_ಅಡ್ಡಸಾಲುಗಳು == 2 ? 2 : ((n_ಅಡ್ಡಸಾಲುಗಳು <= 4 || reduce_ಖಂಡ_ಅಡ್ಡಸಾಲುಗಳು) ? 4 : 8);
        } ಇಲ್ಲದಿದ್ದರೆ {
            ಫಲಿತಾಂಶ.ಖಂಡ_ಅಡ್ಡಸಾಲುಗಳು = n_ಅಡ್ಡಸಾಲುಗಳು <= 4 ? 4 : ((n_ಅಡ್ಡಸಾಲುಗಳು <= 8 || reduce_ಖಂಡ_ಅಡ್ಡಸಾಲುಗಳು) ? 8 : 16);
        }

        ಫಲಿತಾಂಶ.ಖಂಡ_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ = (D & 8) ? 64 : 32;
    }

    ಸ್ಥಿರಾಂಕ uint32_t D_lsb = D ^ (D & (D-1));  // extract lowest set bit

    ಫಲಿತಾಂಶ.d_ವಿಭಜಿಸು = std::min(std::min(ಫಲಿತಾಂಶ.subgroup_ಗಾತ್ರ, 8u), D_lsb / 4);

    ಫಲಿತಾಂಶ.shmem_staging = (device->vendor_id == VK_VENDOR_ID_NVIDIA && hsk < 256 && hsv < 256) ? 1 : 0;

    ಆದರೆ (!reduce_ಖಂಡ_ಅಡ್ಡಸಾಲುಗಳು && !ggml_vk_flash_attn_scalar_shmem_support(device, ಫಲಿತಾಂಶ, hsk, hsv, f32acc, k_type, v_type)) {
        ಫಲಿತಾಂಶ.ಖಂಡ_ಅಡ್ಡಸಾಲುಗಳು /= 2;
    }

    // On AMD RDNA, for small head sizes and big batch size the shader uses few registers, so too many subgroups get scheduled
    // at once and end up thrashing the cache. Fix this by setting a large (unused) shmem buffer that reduces occupancy.
    // This targets an occupancy of 4 subgroups per SIMD.
    ಆದರೆ (device->vendor_id == VK_VENDOR_ID_AMD && device->properties.limits.max_Compute_Shared_ಸ್ಮರಣೆ_ಗಾತ್ರ == 65536) {
        ಆದರೆ (device->architecture != AMD_GCN && n_ಅಡ್ಡಸಾಲುಗಳು >= 64 && hsk <= 128) {
            // 30kb target for hsk > 64, 26kb for <= 64 due to smaller workgroup size
            // Values are guessed, tested on RDNA2
            ಫಲಿತಾಂಶ.ಎಲ್ಲೆ_occupancy_shmem = (hsk <= 64 ? 26 : 30) * 1024 / 4 / 4;
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (device->architecture == AMD_GCN && n_ಅಡ್ಡಸಾಲುಗಳು <= 8 && hsk >= 256) {
            // Same thing for GCN, with an occupancy target of 2 subgroups per SIMD.
            // Here low-batch FA with large head size is affected.
            // n_rows < 4 switch because workgroup size switches from 128 to 256 there.
            ಫಲಿತಾಂಶ.ಎಲ್ಲೆ_occupancy_shmem = (n_ಅಡ್ಡಸಾಲುಗಳು < 4 ? 14 : 26) * 1024 / 4 / 4;
        }
    }

    ಹಿಂತಿರುಗಿ ಫಲಿತಾಂಶ;
}

ಸ್ಥಿರ vk_fa_tuning_params get_fa_tuning_params_coopmat1(ಸ್ಥಿರಾಂಕ vk_device& device, uint32_t hsk, uint32_t hsv, uint32_t n_ಅಡ್ಡಸಾಲುಗಳು, uint32_t n_kv, ggml_type k_type, ggml_type v_type, ತಾರ್ಕಿಕ f32acc) {
    GGML_ಬಳಸದ(n_ಅಡ್ಡಸಾಲುಗಳು);
    GGML_ಬಳಸದ(n_kv);
    GGML_ಬಳಸದ(k_type);
    GGML_ಬಳಸದ(v_type);
    GGML_ಬಳಸದ(f32acc);

    vk_fa_tuning_params ಫಲಿತಾಂಶ{};
    ಫಲಿತಾಂಶ.path = FA_COOPMAT1;

    ಸ್ಥಿರಾಂಕ uint32_t D = hsk | hsv;

    ಸ್ಥಿರಾಂಕ uint32_t coopmat_ಖಂಡ_ಅಡ್ಡಸಾಲುಗಳು = 16;
    ಸ್ಥಿರಾಂಕ uint32_t coopmat_ಖಂಡ_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ = 16;

    ಸ್ಥಿರಾಂಕ uint32_t ಅಂಕೆ_subgroups = 4;

    ಫಲಿತಾಂಶ.ಖಂಡ_ಅಡ್ಡಸಾಲುಗಳು = coopmat_ಖಂಡ_ಅಡ್ಡಸಾಲುಗಳು;
    ಫಲಿತಾಂಶ.ಖಂಡ_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ = coopmat_ಖಂಡ_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ * ಅಂಕೆ_subgroups;
    ಫಲಿತಾಂಶ.ಅಡ್ಡಸಾಲು_ವಿಭಜಿಸು = ಅಂಕೆ_subgroups;
    ಫಲಿತಾಂಶ.subgroup_ಗಾತ್ರ = device->subgroup_ಗಾತ್ರ;
    ಫಲಿತಾಂಶ.workgroup_ಗಾತ್ರ = ಅಂಕೆ_subgroups * ಫಲಿತಾಂಶ.subgroup_ಗಾತ್ರ;

    ಸ್ಥಿರಾಂಕ uint32_t D_lsb = D ^ (D & (D-1));  // extract lowest set bit
    ಫಲಿತಾಂಶ.d_ವಿಭಜಿಸು = std::min(std::min(ಫಲಿತಾಂಶ.subgroup_ಗಾತ್ರ, 8u), D_lsb / 4);

    ಫಲಿತಾಂಶ.shmem_staging = (device->vendor_id == VK_VENDOR_ID_NVIDIA && hsk < 256 && hsv < 256) ? 1 : 0;

    ಹಿಂತಿರುಗಿ ಫಲಿತಾಂಶ;
}

ಸ್ಥಿರ vk_fa_tuning_params get_fa_tuning_params_coopmat2(ಸ್ಥಿರಾಂಕ vk_device& device, uint32_t hsk, uint32_t hsv, uint32_t n_ಅಡ್ಡಸಾಲುಗಳು, uint32_t n_kv, ggml_type k_type, ggml_type v_type, ತಾರ್ಕಿಕ f32acc) {
    GGML_ಬಳಸದ(n_kv);
    GGML_ಬಳಸದ(f32acc);

    vk_fa_tuning_params ಫಲಿತಾಂಶ{};
    ಫಲಿತಾಂಶ.path = FA_COOPMAT2;

    ಸ್ಥಿರಾಂಕ uint32_t D = hsk | hsv;

    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ small_ಅಡ್ಡಸಾಲುಗಳು = n_ಅಡ್ಡಸಾಲುಗಳು < 32;

    ಆದರೆ (small_ಅಡ್ಡಸಾಲುಗಳು) {
        ಫಲಿತಾಂಶ.ಖಂಡ_ಅಡ್ಡಸಾಲುಗಳು = 32;
        ಫಲಿತಾಂಶ.ಖಂಡ_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ = 32;
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (ggml_is_quantized(k_type) || ggml_is_quantized(v_type) || hsk >= 256 || hsv >= 256) {
        ಫಲಿತಾಂಶ.ಖಂಡ_ಅಡ್ಡಸಾಲುಗಳು = (hsk >= 512 || hsv >= 512) ? 32 : 64;
        ಫಲಿತಾಂಶ.ಖಂಡ_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ = 32;
    } ಇಲ್ಲದಿದ್ದರೆ {
        ಫಲಿತಾಂಶ.ಖಂಡ_ಅಡ್ಡಸಾಲುಗಳು = 64;
        ಫಲಿತಾಂಶ.ಖಂಡ_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ = 64;
    }

    ಫಲಿತಾಂಶ.subgroup_ಗಾತ್ರ = device->subgroup_ಗಾತ್ರ;
    ಫಲಿತಾಂಶ.workgroup_ಗಾತ್ರ = (small_ಅಡ್ಡಸಾಲುಗಳು && (D % 32) == 0) ? 256 : 128;

    ಹಿಂತಿರುಗಿ ಫಲಿತಾಂಶ;
}

vk_fa_tuning_params get_fa_tuning_params(ಸ್ಥಿರಾಂಕ vk_device& device, uint32_t hsk, uint32_t hsv, uint32_t n_ಅಡ್ಡಸಾಲುಗಳು, uint32_t n_kv, ggml_type k_type, ggml_type v_type, ತಾರ್ಕಿಕ f32acc) {
    Fa_ಸಂಕೇತ_ಮಾರ್ಗ path = device->coopmat2 ? FA_COOPMAT2 :
                      device->coopmat1_fa_support ? FA_COOPMAT1 : FA_SCALAR;

    ಆದರೆ (path == FA_COOPMAT2 && k_type == GGML_TYPE_BF16 && !device->coopmat2_bf16_support) {
        path = FA_COOPMAT1;
    }
    ಆದರೆ (path == FA_COOPMAT1 && k_type == GGML_TYPE_BF16 && !device->coopmat_bf16_support) {
        path = FA_SCALAR;
    }

    ಆದರೆ (path == FA_COOPMAT1 && device->architecture == vk_device_architecture::NVIDIA_TURING) {
        // Nvidia compiler bug, see https://github.com/ggml-org/llama.cpp/pull/19075#issuecomment-3820716090
        path = FA_SCALAR;
    }

    ಆದರೆ (path == FA_COOPMAT1) {
        ತಾರ್ಕಿಕ shape_ok = (f32acc && device->coopmat_support_16x16x16_f32acc) ||
                        (!f32acc && device->coopmat_support_16x16x16_f16acc);
        ಸ್ಥಿರಾಂಕ vk_fa_tuning_params params = get_fa_tuning_params_coopmat1(device, hsk, hsv, n_ಅಡ್ಡಸಾಲುಗಳು, n_kv, k_type, v_type, f32acc);
        ತಾರ್ಕಿಕ shmem_ok = ggml_vk_flash_attn_coopmat_shmem_support(device, params, hsk, hsv, f32acc, k_type, v_type);

        ಆದರೆ (!shape_ok || !shmem_ok) {
            path = FA_SCALAR;
        }
    }

    // scalar is faster than coopmat when N==1
    ಆದರೆ (n_ಅಡ್ಡಸಾಲುಗಳು == 1 && (path == FA_COOPMAT1 || path == FA_COOPMAT2)) {
        path = FA_SCALAR;
    }

    ಆಯ್ಕೆ (path) {
    ಸ್ಥಿತಿ FA_SCALAR:
        ಹಿಂತಿರುಗಿ get_fa_tuning_params_scalar(device, hsk, hsv, n_ಅಡ್ಡಸಾಲುಗಳು, n_kv, k_type, v_type, f32acc);
    ಸ್ಥಿತಿ FA_COOPMAT1:
        ಹಿಂತಿರುಗಿ get_fa_tuning_params_coopmat1(device, hsk, hsv, n_ಅಡ್ಡಸಾಲುಗಳು, n_kv, k_type, v_type, f32acc);
    ಸ್ಥಿತಿ FA_COOPMAT2:
        ಹಿಂತಿರುಗಿ get_fa_tuning_params_coopmat2(device, hsk, hsv, n_ಅಡ್ಡಸಾಲುಗಳು, n_kv, k_type, v_type, f32acc);
    ಪೂರ್ವನಿಯೋಜಿತ:
        ಎಸೆ std::runtime_error("unsupported FaCodePath");
    }
}

vk_fa_pipeline_state get_fa_pipeline_state(ಸ್ಥಿರಾಂಕ vk_device& device, ಸ್ಥಿರಾಂಕ vk_fa_tuning_params& params, uint32_t hsk, uint32_t hsv, ತಾರ್ಕಿಕ aligned, ತಾರ್ಕಿಕ f32acc,
                                                  ತಾರ್ಕಿಕ use_ಮುಖವಾಡ, ತಾರ್ಕಿಕ use_ಮುಖವಾಡ_opt, ತಾರ್ಕಿಕ use_logit_softcap, ತಾರ್ಕಿಕ use_sparse, ggml_type k_type, ggml_type v_type) {
    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ old_amd_windows = device->vendor_id == VK_VENDOR_ID_AMD && device->driver_id == vk::DriverId::eAmdProprietary &&
                                 (device->architecture == AMD_GCN || device->architecture == AMD_RDNA1 || device->architecture == AMD_RDNA2);

    uint32_t flags = (use_ಮುಖವಾಡ_opt      ? 1 : 0) |
                     (use_ಮುಖವಾಡ          ? 2 : 0) |
                     (use_logit_softcap ? 4 : 0) |
                     (old_amd_windows   ? 8 : 0) |
                     (use_sparse        ? 16 : 0);

    ಸ್ಥಿರಾಂಕ uint32_t subgroup_ಗಾತ್ರ = params.disable_subgroups ? 0 : params.subgroup_ಗಾತ್ರ;

    ಹಿಂತಿರುಗಿ vk_fa_pipeline_state{hsk, hsv, params.ಖಂಡ_ಅಡ್ಡಸಾಲುಗಳು, params.ಖಂಡ_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ, params.d_ವಿಭಜಿಸು, params.ಅಡ್ಡಸಾಲು_ವಿಭಜಿಸು, params.shmem_staging, params.path, params.workgroup_ಗಾತ್ರ, subgroup_ಗಾತ್ರ, aligned, f32acc, flags, params.ಎಲ್ಲೆ_occupancy_shmem, k_type, v_type};
}

ಸ್ಥಿರ uint32_t fa_ಖಂಡ_ಬೈಟುಗಳು(ggml_type t) {
    ಆದರೆ (t == GGML_TYPE_F32) {
        ಹಿಂತಿರುಗಿ 16u;
    }
    ಹಿಂತಿರುಗಿ (uint32_t) ggml_type_ಗಾತ್ರ(t);
}

ಸ್ಥಿರ std::vector<uint32_t> get_fa_spec_constants(ಸ್ಥಿರಾಂಕ vk_fa_pipeline_state& state) {
    ಹಿಂತಿರುಗಿ {
        /* 0 WorkGroupSize   */ state.workgroup_ಗಾತ್ರ,
        /* 1 Br              */ state.Br,
        /* 2 Bc              */ state.Bc,
        /* 3 HSK             */ state.HSK,
        /* 4 HSV             */ state.HSV,
        /* 5 Clamp           */ static_cast<uint32_t>(!state.aligned),
        /* 6 D_split         */ state.D_ವಿಭಜಿಸು,
        /* 7 row_split       */ state.ಅಡ್ಡಸಾಲು_ವಿಭಜಿಸು,
        /* 8 SubGroupSize    */ state.subgroup_ಗಾತ್ರ,
        /* 9 SHMEM_STAGING   */ state.shmem_staging ? 1u : 0u,
        /*10 Flags           */ state.flags,
        /*11 LIMIT_OCCUPANCY_SHMEM */ state.ಎಲ್ಲೆ_occupancy_shmem,
        /*12 FaTypeK         */ static_cast<uint32_t>(state.k_type),
        /*13 FaTypeV         */ static_cast<uint32_t>(state.v_type),
        /*14 FaBlockBytesK   */ fa_ಖಂಡ_ಬೈಟುಗಳು(state.k_type),
        /*15 FaBlockBytesV   */ fa_ಖಂಡ_ಬೈಟುಗಳು(state.v_type),
    };
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_vk_matmul_shmem_support(ಸ್ಥಿರಾಂಕ vk_device& device, ಸ್ಥಿರಾಂಕ std::vector<uint32_t>& warptile, ತಾರ್ಕಿಕ mul_mat_id, ggml_type src0_type) {

    uint32_t lut_ಗಾತ್ರ = 0;
    ಆಯ್ಕೆ (src0_type) {
    ಸ್ಥಿತಿ GGML_TYPE_IQ1_S:
    ಸ್ಥಿತಿ GGML_TYPE_IQ1_M:
        // Regular matmul uses the compact uint16_t IQ1 grid; the expanded
        // uint32_t grid is only enabled for the q8_1/int-dot vector path.
        lut_ಗಾತ್ರ = 2*2048;
        ನಿಲ್ಲಿಸು;
    ಸ್ಥಿತಿ GGML_TYPE_IQ2_XXS:
        lut_ಗಾತ್ರ = 8*256;
        ನಿಲ್ಲಿಸು;
    ಸ್ಥಿತಿ GGML_TYPE_IQ2_XS:
        lut_ಗಾತ್ರ = 8*512;
        ನಿಲ್ಲಿಸು;
    ಸ್ಥಿತಿ GGML_TYPE_IQ2_S:
        lut_ಗಾತ್ರ = 8*1024;
        ನಿಲ್ಲಿಸು;
    ಸ್ಥಿತಿ GGML_TYPE_IQ3_XXS:
        lut_ಗಾತ್ರ = 4*256;
        ನಿಲ್ಲಿಸು;
    ಸ್ಥಿತಿ GGML_TYPE_IQ3_S:
        lut_ಗಾತ್ರ = 4*512;
        ನಿಲ್ಲಿಸು;
    ಸ್ಥಿತಿ GGML_TYPE_IQ4_NL:
    ಸ್ಥಿತಿ GGML_TYPE_IQ4_XS:
    ಸ್ಥಿತಿ GGML_TYPE_MXFP4:
        lut_ಗಾತ್ರ = 4*16;
        ನಿಲ್ಲಿಸು;
    ಸ್ಥಿತಿ GGML_TYPE_NVFP4:
        // Same kvalues budget as MXFP4 plus ue4m3_fp32_lut[128] (types.glsl, DATA_A_NVFP4).
        lut_ಗಾತ್ರ = 4*16 + 128u * (uint32_t)ಗಾತ್ರ(ದಶಮಾಂಶ);
        ನಿಲ್ಲಿಸು;
    ಪೂರ್ವನಿಯೋಜಿತ:
        ನಿಲ್ಲಿಸು;
    }

    // Needs to be kept up to date on shader changes
    // Needs to stay aligned with ggml_vk_mul_mm_spec.
    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ intel_shmem_stride_pad_zero = device->vendor_id == VK_VENDOR_ID_INTEL && device->coopmat_support &&
                                              device->driver_id == vk::DriverId::eIntelProprietaryWindows;
    ಸ್ಥಿರಾಂಕ uint32_t bank_conflict_ಸ್ಥಾನಾಂತರ = intel_shmem_stride_pad_zero ? 0 : (device->coopmat_support ? 8 : 1);
    ಸ್ಥಿರಾಂಕ uint32_t type_ಗಾತ್ರ = device->fp16 ? ಗಾತ್ರ(ggml_fp16_t) : ಗಾತ್ರ(ದಶಮಾಂಶ);
    ಸ್ಥಿರಾಂಕ uint32_t warps = warptile[0] / warptile[10];

    ಸ್ಥಿರಾಂಕ uint32_t ತುಂಬು_bufs = (warptile[1] + warptile[2]) * (warptile[3] + bank_conflict_ಸ್ಥಾನಾಂತರ) * type_ಗಾತ್ರ;
    ಸ್ಥಿರಾಂಕ uint32_t mmid_ಅಡ್ಡಸಾಲು_ids = mul_mat_id ? (warptile[2] * 2 * ಗಾತ್ರ(uint16_t)) : 0;
    ಸ್ಥಿರಾಂಕ uint32_t coopmat_stage = device->coopmat_support ? warptile[7] * warptile[8] / warps * ಗಾತ್ರ(ದಶಮಾಂಶ) : 0;

    ಸ್ಥಿರಾಂಕ uint32_t ಒಟ್ಟು_ಗಾತ್ರ = ತುಂಬು_bufs + mmid_ಅಡ್ಡಸಾಲು_ids + coopmat_stage + lut_ಗಾತ್ರ;
    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ supported = ಒಟ್ಟು_ಗಾತ್ರ <= device->properties.limits.max_Compute_Shared_ಸ್ಮರಣೆ_ಗಾತ್ರ;

    VK_LOG_DEBUG("ggml_vk_matmul_shmem_support(warptile=(" << warptile[0] << "," << warptile[1] << "," << warptile[2] << "), "
                 "mul_mat_id=" << mul_mat_id << ", src0_type=" << ggml_type_name(src0_type) << ", supported=" << supported);

    ಹಿಂತಿರುಗಿ supported;
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_vk_matmul_int_shmem_support(ಸ್ಥಿರಾಂಕ vk_device& device, ಸ್ಥಿರಾಂಕ std::vector<uint32_t>& warptile, ತಾರ್ಕಿಕ mul_mat_id, ggml_type src0_type) {

    // FLOAT_TYPE in the shader is float16_t with fp16 support, otherwise float.
    ಸ್ಥಿರಾಂಕ uint32_t fp_ಗಾತ್ರ   = device->fp16 ? 2u : 4u;
    ಸ್ಥಿರಾಂಕ uint32_t fp_align  = fp_ಗಾತ್ರ;
    ಸ್ಥಿರಾಂಕ uint32_t fp2_ಗಾತ್ರ  = 2u * fp_ಗಾತ್ರ;
    ಸ್ಥಿರಾಂಕ uint32_t fp2_align = device->fp16 ? 4u : 8u;

    ರಚನೆ member { uint32_t size, align; };
    ಸ್ವಯಂಚಾಲಿತ std430_ಗಾತ್ರ = [](std::initializer_list<member> members) {
        uint32_t off = 0, struct_align = 1;
        ಗಾಗಿ (ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ &m : members) {
            off = (off + m.align - 1) & ~(m.align - 1);
            off += m.size;
            struct_align = std::max(struct_align, m.align);
        }
        ಹಿಂತಿರುಗಿ (off + struct_align - 1) & ~(struct_align - 1);
    };

    uint32_t ಖಂಡ_a_ಗಾತ್ರ = 0;
    ಆಯ್ಕೆ (src0_type) {
        ಸ್ಥಿತಿ GGML_TYPE_Q2_0:    ಖಂಡ_a_ಗಾತ್ರ = std430_ಗಾತ್ರ({{32, 4}, {fp_ಗಾತ್ರ,  fp_align}});                  ನಿಲ್ಲಿಸು; // qs[8] + dm
        ಸ್ಥಿತಿ GGML_TYPE_Q4_0:    ಖಂಡ_a_ಗಾತ್ರ = std430_ಗಾತ್ರ({{16, 4}, {fp_ಗಾತ್ರ,  fp_align}});                  ನಿಲ್ಲಿಸು; // qs[16/4] + dm
        ಸ್ಥಿತಿ GGML_TYPE_Q4_1:    ಖಂಡ_a_ಗಾತ್ರ = std430_ಗಾತ್ರ({{16, 4}, {fp2_ಗಾತ್ರ, fp2_align}});                 ನಿಲ್ಲಿಸು; // qs[16/4] + dm(vec2)
        ಸ್ಥಿತಿ GGML_TYPE_Q5_0:    ಖಂಡ_a_ಗಾತ್ರ = std430_ಗಾತ್ರ({{16, 4}, {4, 4}, {fp_ಗಾತ್ರ,  fp_align}});          ನಿಲ್ಲಿಸು; // qs[16/4] + qh + dm
        ಸ್ಥಿತಿ GGML_TYPE_Q5_1:    ಖಂಡ_a_ಗಾತ್ರ = std430_ಗಾತ್ರ({{16, 4}, {4, 4}, {fp2_ಗಾತ್ರ, fp2_align}});         ನಿಲ್ಲಿಸು; // qs[16/4] + qh + dm(vec2)
        ಸ್ಥಿತಿ GGML_TYPE_Q8_0:    ಖಂಡ_a_ಗಾತ್ರ = std430_ಗಾತ್ರ({{32, 4}, {fp_ಗಾತ್ರ,  fp_align}});                  ನಿಲ್ಲಿಸು; // qs[8] + dm
        ಸ್ಥಿತಿ GGML_TYPE_IQ4_XS:  ಖಂಡ_a_ಗಾತ್ರ = std430_ಗಾತ್ರ({{32, 4}, {fp_ಗಾತ್ರ,  fp_align}});                  ನಿಲ್ಲಿಸು; // qs[8] + d
        ಸ್ಥಿತಿ GGML_TYPE_MXFP4:   ಖಂಡ_a_ಗಾತ್ರ = std430_ಗಾತ್ರ({{32, 4}, {fp_ಗಾತ್ರ,  fp_align}});                  ನಿಲ್ಲಿಸು; // qs[8] + d
        ಸ್ಥಿತಿ GGML_TYPE_IQ4_NL:  ಖಂಡ_a_ಗಾತ್ರ = std430_ಗಾತ್ರ({{32, 4}, {fp_ಗಾತ್ರ,  fp_align}});                  ನಿಲ್ಲಿಸು; // qs[8] + d
        ಸ್ಥಿತಿ GGML_TYPE_NVFP4:   ಖಂಡ_a_ಗಾತ್ರ = std430_ಗಾತ್ರ({{32, 4}, {fp2_ಗಾತ್ರ, fp2_align}});                 ನಿಲ್ಲಿಸು; // qs[8] + d_scales(vec2)
        ಸ್ಥಿತಿ GGML_TYPE_Q2_K:    ಖಂಡ_a_ಗಾತ್ರ = std430_ಗಾತ್ರ({{ 8, 4}, {2, 2}, {fp2_ಗಾತ್ರ, fp2_align}});         ನಿಲ್ಲಿಸು; // qs[2] + scales(u8vec2) + dm(vec2)
        ಸ್ಥಿತಿ GGML_TYPE_Q3_K:    ಖಂಡ_a_ಗಾತ್ರ = std430_ಗಾತ್ರ({{16, 4}, {fp2_ಗಾತ್ರ, fp2_align}});                 ನಿಲ್ಲಿಸು; // qs[4] + d_scales(vec2)
        ಸ್ಥಿತಿ GGML_TYPE_Q4_K:    ಖಂಡ_a_ಗಾತ್ರ = std430_ಗಾತ್ರ({{16, 4}, {fp2_ಗಾತ್ರ, fp2_align}});                 ನಿಲ್ಲಿಸು; // qs[4] + dm(vec2)
        ಸ್ಥಿತಿ GGML_TYPE_Q5_K:    ಖಂಡ_a_ಗಾತ್ರ = std430_ಗಾತ್ರ({{32, 4}, {fp2_ಗಾತ್ರ, fp2_align}});                 ನಿಲ್ಲಿಸು; // qs[8] + dm(vec2)
        ಸ್ಥಿತಿ GGML_TYPE_Q6_K:    ಖಂಡ_a_ಗಾತ್ರ = std430_ಗಾತ್ರ({{32, 4}, {fp2_ಗಾತ್ರ, fp2_align}});                 ನಿಲ್ಲಿಸು; // qs[8] + d_scales(vec2)
        ಸ್ಥಿತಿ GGML_TYPE_IQ3_S:   ಖಂಡ_a_ಗಾತ್ರ = std430_ಗಾತ್ರ({{32, 4}, {fp_ಗಾತ್ರ,  fp_align}});                  ನಿಲ್ಲಿಸು; // qs[8] + d
        ಪೂರ್ವನಿಯೋಜಿತ:
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    // IQ3_S also copies its 512-entry grid into shared memory (types.glsl, init_iq_shmem)
    ಸ್ಥಿರಾಂಕ uint32_t lut_ಗಾತ್ರ = (src0_type == GGML_TYPE_IQ3_S) ? 4*512 : 0;

    // block_b_cache: { int32_t qs[8]; FLOAT_TYPEV2 ds; }
    ಸ್ಥಿರಾಂಕ uint32_t ಖಂಡ_b_ಗಾತ್ರ = std430_ಗಾತ್ರ({{32, 4}, {fp2_ಗಾತ್ರ, fp2_align}});

    ಸ್ಥಿರಾಂಕ uint32_t BM = warptile[1];
    ಸ್ಥಿರಾಂಕ uint32_t BN = warptile[2];
    // mul_mmq.comp: BK_STEP=1 for MUL_MAT_ID, 4 otherwise.
    ಸ್ಥಿರಾಂಕ uint32_t BK_ಹೆಜ್ಜೆ = mul_mat_id ? 1u : 4u;

    ಸ್ಥಿರಾಂಕ uint32_t buf_a_ಗಾತ್ರ = BM * BK_ಹೆಜ್ಜೆ * ಖಂಡ_a_ಗಾತ್ರ;
    ಸ್ಥಿರಾಂಕ uint32_t buf_b_ಗಾತ್ರ = BN * BK_ಹೆಜ್ಜೆ * ಖಂಡ_b_ಗಾತ್ರ;
    ಸ್ಥಿರಾಂಕ uint32_t mmid_ಅಡ್ಡಸಾಲು_ids = mul_mat_id ? (BN * 2u * (uint32_t)ಗಾತ್ರ(uint16_t)) : 0u;

    ಸ್ಥಿರಾಂಕ uint32_t ಒಟ್ಟು_ಗಾತ್ರ = buf_a_ಗಾತ್ರ + buf_b_ಗಾತ್ರ + mmid_ಅಡ್ಡಸಾಲು_ids + lut_ಗಾತ್ರ;
    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ supported = ಒಟ್ಟು_ಗಾತ್ರ <= device->properties.limits.max_Compute_Shared_ಸ್ಮರಣೆ_ಗಾತ್ರ;

    VK_LOG_DEBUG("ggml_vk_matmul_int_shmem_support(warptile=(" << warptile[0] << "," << warptile[1] << "," << warptile[2] << "), "
                 "mul_mat_id=" << mul_mat_id << ", src0_type=" << ggml_type_name(src0_type) << ", total=" << ಒಟ್ಟು_ಗಾತ್ರ << ", supported=" << supported);

    ಹಿಂತಿರುಗಿ supported;
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_vk_matmul_cm1_int_shmem_support(ಸ್ಥಿರಾಂಕ vk_device& device, ಸ್ಥಿರಾಂಕ std::vector<uint32_t>& warptile, ತಾರ್ಕಿಕ mul_mat_id, ggml_type src0_type) {

    ತಾರ್ಕಿಕ kscales2 = ಸುಳ್ಳು;    // two scale sets per block
    ತಾರ್ಕಿಕ has_dm   = ಸುಳ್ಳು;    // d+m as vec2 + b-side sum
    ತಾರ್ಕಿಕ has_kvalues = ಸುಳ್ಳು;
    ಆಯ್ಕೆ (src0_type) {
        ಸ್ಥಿತಿ GGML_TYPE_Q4_0: ಸ್ಥಿತಿ GGML_TYPE_Q5_0: ಸ್ಥಿತಿ GGML_TYPE_Q8_0:
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_Q4_1: ಸ್ಥಿತಿ GGML_TYPE_Q5_1:
        ಸ್ಥಿತಿ GGML_TYPE_Q4_K: ಸ್ಥಿತಿ GGML_TYPE_Q5_K:
            has_dm = ನಿಜ;                          ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_IQ4_NL: ಸ್ಥಿತಿ GGML_TYPE_IQ4_XS: ಸ್ಥಿತಿ GGML_TYPE_MXFP4:
            has_kvalues = ನಿಜ;                     ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_Q3_K: ಸ್ಥಿತಿ GGML_TYPE_Q6_K:
            kscales2 = ನಿಜ;                        ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_NVFP4:
            kscales2 = ನಿಜ; has_kvalues = ನಿಜ;    ನಿಲ್ಲಿಸು;
        ಪೂರ್ವನಿಯೋಜಿತ:
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಸ್ಥಿರಾಂಕ uint32_t BM         = warptile[1];
    ಸ್ಥಿರಾಂಕ uint32_t BN         = warptile[2];

    ಸ್ಥಿರಾಂಕ uint32_t BK      = 32;
    ಸ್ಥಿರಾಂಕ uint32_t BK_ಹೆಜ್ಜೆ = mul_mat_id ? 2u : 4u;
    ಸ್ಥಿರಾಂಕ uint32_t QPITCH  = BK_ಹೆಜ್ಜೆ * (BK / 4u) + 4u;
    ಸ್ಥಿರಾಂಕ uint32_t KSCALES = kscales2 ? 2u : 1u;

    uint32_t ಒಟ್ಟು = 0;
    ಒಟ್ಟು += BM * QPITCH * (uint32_t)ಗಾತ್ರ(uint32_t);   // buf_a_qs
    ಒಟ್ಟು += BN * QPITCH * (uint32_t)ಗಾತ್ರ(uint32_t);   // buf_b_qs
    ಒಟ್ಟು += has_dm ? (BM * BK_ಹೆಜ್ಜೆ * 2u * (uint32_t)ಗಾತ್ರ(ದಶಮಾಂಶ))   // buf_a_dm (vec2)
                    : (BM * BK_ಹೆಜ್ಜೆ * KSCALES * (uint32_t)ಗಾತ್ರ(ದಶಮಾಂಶ)); // buf_a_d
    ಒಟ್ಟು += BN * BK_ಹೆಜ್ಜೆ * (uint32_t)ಗಾತ್ರ(ದಶಮಾಂಶ);     // buf_b_d
    ಆದರೆ (has_dm) {
        ಒಟ್ಟು += BN * BK_ಹೆಜ್ಜೆ * (uint32_t)ಗಾತ್ರ(ದಶಮಾಂಶ); // buf_b_s
    }
    ಆದರೆ (has_kvalues) {
        ಒಟ್ಟು += 16u * (uint32_t)ಗಾತ್ರ(int8_t);         // cm1_kvalues[16]
    }
    ಆದರೆ (src0_type == GGML_TYPE_NVFP4 && !device->ocp_fp4) {
        ಒಟ್ಟು += 128u * (uint32_t)ಗಾತ್ರ(ದಶಮಾಂಶ);         // ue4m3_fp32_lut[128]
    }
    ಆದರೆ (mul_mat_id) {
        ಒಟ್ಟು += BN * 2u * (uint32_t)ಗಾತ್ರ(uint16_t);   // row_ids[BN] (u16vec2)
    }

    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ supported = ಒಟ್ಟು <= device->properties.limits.max_Compute_Shared_ಸ್ಮರಣೆ_ಗಾತ್ರ;

    VK_LOG_DEBUG("ggml_vk_matmul_cm1_int_shmem_support(warptile=(" << warptile[0] << "," << warptile[1] << "," << warptile[2] << "), "
                 "mul_mat_id=" << mul_mat_id << ", src0_type=" << ggml_type_name(src0_type) << ", total=" << ಒಟ್ಟು << ", supported=" << supported);

    ಹಿಂತಿರುಗಿ supported;
}

ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ std::unordered_map<std::string, uint32_t> rdna1_pipelines = {
    {"soft_max", 64}, {"im2col", 64},
    {"argmax", 64}, {"mul_mat_vec", 64},
    {"mul_mat_vec_f16", 32}, {"mul_mat_vec_f32_f16", 32}
};

ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ std::unordered_map<std::string, uint32_t> rdna2_pipelines = {
    {"soft_max", 64}, {"im2col", 64},
};

ಸ್ಥಿರ std::vector<GpuPipelineConfig> gpu_pipeline_configs = {
    {
        vk_device_architecture::AMD_RDNA1,
        {
            rdna1_pipelines,
        },
        RDNA_DEFAULT_SUBGROUP_ಗಾತ್ರ
    },
    {
        vk_device_architecture::AMD_RDNA2,
        {
            rdna2_pipelines,
        },
        RDNA_DEFAULT_SUBGROUP_ಗಾತ್ರ
    },
};

uint32_t get_subgroup_ಗಾತ್ರ(ಸ್ಥಿರಾಂಕ std::string &pipeline_name, ಸ್ಥಿರಾಂಕ vk_device_architecture &arch) {
    ಗಾಗಿ (ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ &config : gpu_pipeline_configs) {
        ಆದರೆ (config.arch == arch) {
            ಸ್ವಯಂಚಾಲಿತ pipIt = config.pipelines.find(pipeline_name);
            ಆದರೆ (pipIt != config.pipelines.end()) {
                ಹಿಂತಿರುಗಿ pipIt->second;
            }
            std::vector<std::pair<std::string, uint32_t>> sorted_pipelines(config.pipelines.begin(), config.pipelines.end());
            std::sort(sorted_pipelines.begin(), sorted_pipelines.end(),
                      [](ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ &a, ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ &b) { ಹಿಂತಿರುಗಿ a.ಮೊದಲ.size() > b.ಮೊದಲ.size(); });
            ಗಾಗಿ (ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ &ನಮೂದು : sorted_pipelines) {
                ಆದರೆ (pipeline_name.find(ನಮೂದು.ಮೊದಲ) != std::string::npos) {
                    ಹಿಂತಿರುಗಿ ನಮೂದು.second;
                }
            }
            ಹಿಂತಿರುಗಿ config.default_subgroup_ಗಾತ್ರ;
        }
    }
    ಹಿಂತಿರುಗಿ 0; // If no matching configuration is found
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_vk_fa_type_needs_shmem(ggml_type type) {
    ಆಯ್ಕೆ (type) {
    ಸ್ಥಿತಿ GGML_TYPE_IQ4_NL:
        ಹಿಂತಿರುಗಿ ನಿಜ;
    ಪೂರ್ವನಿಯೋಜಿತ:
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_vk_fa_scalar_ಬಳಸುತ್ತದೆ_mmq(ಸ್ಥಿರಾಂಕ vk_device& device, ggml_type k_type, ggml_type v_type) {
#ಆದರೆ defined(GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT)
    ಹಿಂತಿರುಗಿ device->integer_dot_ಗುಣಲಬ್ಧ && device->subgroup_clustered &&
           !ggml_vk_fa_type_needs_shmem(v_type) &&
           (k_type == GGML_TYPE_Q4_0 || k_type == GGML_TYPE_Q4_1 ||
            k_type == GGML_TYPE_Q5_0 || k_type == GGML_TYPE_Q5_1 ||
            k_type == GGML_TYPE_Q8_0);
#ಇಲ್ಲದಿದ್ದರೆ
    GGML_ಬಳಸದ(device);
    GGML_ಬಳಸದ(k_type);
    GGML_ಬಳಸದ(v_type);
    ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
#ಆದರೆ_ಅಂತ್ಯ
}

ಶೂನ್ಯ ggml_vk_ತುಂಬು_shaders(vk_device& device, vk_pipeline requested) {
    VK_LOG_DEBUG("ggml_vk_load_shaders(" << device->name << ")");

    // some shaders have a minimum subgroup size
    ಸ್ಥಿರಾಂಕ uint32_t subgroup_ಗಾತ್ರ_8 = std::max(device->subgroup_ಗಾತ್ರ, 8u);
    ಸ್ಥಿರಾಂಕ uint32_t subgroup_ಗಾತ್ರ_16 = std::max(device->subgroup_ಗಾತ್ರ, 16u);
    ಸ್ಥಿರಾಂಕ uint32_t subgroup_ಗಾತ್ರ_32 = std::max(device->subgroup_ಗಾತ್ರ, 32u);

    // clamp WARP for l_/m_ warptiles so WM <= BM (breaks on subgroupSize > 64)
    ಸ್ಥಿರಾಂಕ uint32_t mm_warp_8  = std::min(subgroup_ಗಾತ್ರ_8,  64u);
    ಸ್ಥಿರಾಂಕ uint32_t mm_warp_16 = std::min(subgroup_ಗಾತ್ರ_16, 64u);

    ಸ್ಥಿರಾಂಕ uint32_t mul_mat_subgroup_ಗಾತ್ರ = (device->vendor_id == VK_VENDOR_ID_INTEL && device->subgroup_ಗಾತ್ರ_control) ? device->subgroup_min_ಗಾತ್ರ : device->subgroup_ಗಾತ್ರ;
    ಸ್ಥಿರಾಂಕ uint32_t mul_mat_subgroup_ಗಾತ್ರ_8 = std::max(mul_mat_subgroup_ಗಾತ್ರ, 8u);
    ಸ್ಥಿರಾಂಕ uint32_t mul_mat_subgroup_ಗಾತ್ರ_16 = std::max(mul_mat_subgroup_ಗಾತ್ರ, 16u);
    ಸ್ಥಿರಾಂಕ uint32_t mul_mat_subgroup_ಗಾತ್ರ_32 = std::max(mul_mat_subgroup_ಗಾತ್ರ, 32u);
    ಸ್ಥಿರಾಂಕ uint32_t mul_mat_mm_warp_8  = std::min(mul_mat_subgroup_ಗಾತ್ರ_8,  64u);
    ಸ್ಥಿರಾಂಕ uint32_t mul_mat_mm_warp_16 = std::min(mul_mat_subgroup_ಗಾತ್ರ_16, 64u);

    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ subgroup_min_ಗಾತ್ರ_16 = (!device->subgroup_ಗಾತ್ರ_control && device->subgroup_ಗಾತ್ರ >= 16) ||
                                      (device->subgroup_ಗಾತ್ರ_control && device->subgroup_max_ಗಾತ್ರ >= 16);

    // mulmat
    // Warptile layout (indices match mul_mm.comp constantIDs):
    //   [0..9]  : BLOCK_SIZE, BM, BN, BK, WM, WN, WMITER, TM, TN, TK
    //   [10]    : WARP / required_subgroup_size (read via WARP_SIZE_IDX)
    ಸ್ಥಿರ constexpr size_t WARP_ಗಾತ್ರ_IDX = 10;
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
    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ cm1_use_wave32 = device->vendor_id == VK_VENDOR_ID_AMD &&
                                device->subgroup_ಗಾತ್ರ_control &&
                                device->subgroup_min_ಗಾತ್ರ <= 32 && device->subgroup_max_ಗಾತ್ರ >= 32;
    ಸ್ಥಿರಾಂಕ uint32_t cm1_sg = cm1_use_wave32 ? 32 : device->subgroup_ಗಾತ್ರ;

    vk_pipeline wait_pipeline;
    CompileTask claimed_task {};
    ತಾರ್ಕಿಕ has_claimed_task = ಸುಳ್ಳು;

    // The rest of the walk reads and writes shared device state, so hold the
    // lock until we're done deciding what to compile.
    std::unique_lock<std::mutex> compile_ಬೀಗ(device->compile_mutex);

    ಆದರೆ (device->coopmat2) {
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
        ಸ್ಥಿರಾಂಕ uint32_t mmqid_bk = device->coopmat2_decode_ಸದಿಶ ? 64u : 32u;
        l_warptile_mmqid = { 256, 128, 128, mmqid_bk, 1 };
        m_warptile_mmqid = { 256, 128, 64,  mmqid_bk, 0 };
        s_warptile_mmqid = { 256, 128, 64,  mmqid_bk, 0 };
        l_mmqid_wg_denoms = { 128, 128, 1 };
        m_mmqid_wg_denoms = { 128, 64, 1 };
        s_mmqid_wg_denoms = { 128, 64, 1 };

        l_align = 128;
        m_align =  64;
        s_align =  32;
    } ಇಲ್ಲದಿದ್ದರೆ {
        // Matrix cores require different warp group sizes
        ಸ್ಥಿರಾಂಕ uint32_t tm_l = device->coopmat_support ? device->coopmat_m : 4;
        ಸ್ಥಿರಾಂಕ uint32_t tm_m = device->coopmat_support ? device->coopmat_m : 4;
        ಸ್ಥಿರಾಂಕ uint32_t tm_s = device->coopmat_support ? device->coopmat_m : 2;
        ಸ್ಥಿರಾಂಕ uint32_t tn_l = device->coopmat_support ? device->coopmat_n : 4;
        ಸ್ಥಿರಾಂಕ uint32_t tn_m = device->coopmat_support ? device->coopmat_n : 2;
        ಸ್ಥಿರಾಂಕ uint32_t tn_s = device->coopmat_support ? device->coopmat_n : 2;
        ಸ್ಥಿರಾಂಕ uint32_t tk_l = device->coopmat_support ? device->coopmat_k : 1;
        ಸ್ಥಿರಾಂಕ uint32_t tk_m = device->coopmat_support ? device->coopmat_k : 1;
        ಸ್ಥಿರಾಂಕ uint32_t tk_s = device->coopmat_support ? device->coopmat_k : 1;

        ಸ್ಥಿರಾಂಕ uint32_t itm = device->coopmat_int_m;
        ಸ್ಥಿರಾಂಕ uint32_t itn = device->coopmat_int_n;
        ಸ್ಥಿರಾಂಕ uint32_t itk = device->coopmat_int_k;

        ಸ್ಥಿರಾಂಕ uint32_t s_warptile_wm = device->subgroup_ಗಾತ್ರ == 8 ? 8 : 32;

        l_warptile = { 128,             128, 128, 16, mm_warp_8 * 2, 64, 2, tm_l, tn_l, tk_l, mm_warp_8 };
        m_warptile = { 128,              64,  64, 16, mm_warp_8,     32, 2, tm_m, tn_m, tk_m, mm_warp_8 };
        s_warptile = { subgroup_ಗಾತ್ರ_32, 32,  32, 16, s_warptile_wm, 32, 2, tm_s, tn_s, tk_s, subgroup_ಗಾತ್ರ_8 };

        l_warptile_mmq = { 128,             128, 128, 32, mm_warp_8 * 2, 64, 2, tm_l, tn_l, tk_l, mm_warp_8 };
        m_warptile_mmq = { 128,              64,  64, 32, mm_warp_8,     32, 2, tm_m, tn_m, tk_m, mm_warp_8 };
        s_warptile_mmq = { subgroup_ಗಾತ್ರ_32, 32,  32, 32, s_warptile_wm, 32, 2, tm_s, tn_s, tk_s, subgroup_ಗಾತ್ರ_8 };

        // Integer MMQ has a smaller shared memory profile, but heavier register use
        l_warptile_mmq_int = { 128,             128, 128, 32, mm_warp_8 * 2, 64, 2, 4, 4, 1, mm_warp_8 };
        m_warptile_mmq_int = { 128,              64,  64, 32, mm_warp_8,     32, 2, 2, 2, 1, mm_warp_8 };
        s_warptile_mmq_int = { subgroup_ಗಾತ್ರ_32, 32,  32, 32, s_warptile_wm, 32, 2, 2, 1, 1, subgroup_ಗಾತ್ರ_8 };

        ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ cm1_bs = [cm1_sg](uint32_t bm, uint32_t bn) {
            ಹಿಂತಿರುಗಿ cm1_sg * (bm / std::min(cm1_sg, bm)) * (bn / 32);
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
        s_warptile_mmq_int_k = { subgroup_ಗಾತ್ರ_32,   32,  32, 32, s_warptile_wm, 32, 1, 2, 1, 1, subgroup_ಗಾತ್ರ_8 };

        l_warptile_id = { 128,                      128, 128, 16, mul_mat_mm_warp_16 * 2, 64, 2, tm_l, tn_l, tk_l, mul_mat_mm_warp_16 };
        m_warptile_id = { 128,                       64,  64, 16, mul_mat_mm_warp_16,     32, 2, tm_m, tn_m, tk_m, mul_mat_mm_warp_16 };
        s_warptile_id = { mul_mat_subgroup_ಗಾತ್ರ_16,  32,  32, 16, s_warptile_wm,          32, 2, tm_s, tn_s, tk_s, mul_mat_subgroup_ಗಾತ್ರ_16 };

        l_warptile_mmqid = { 128,                       128, 128, 32, mul_mat_mm_warp_8 * 2, 64, 2, tm_l, tn_l, tk_l, mul_mat_mm_warp_8 };
        m_warptile_mmqid = { 128,                        64,  64, 32, mul_mat_mm_warp_8,     32, 2, tm_m, tn_m, tk_m, mul_mat_mm_warp_8 };
        s_warptile_mmqid = { mul_mat_subgroup_ಗಾತ್ರ_32,   32,  32, 32, s_warptile_wm,         32, 2, tm_s, tn_s, tk_s, mul_mat_subgroup_ಗಾತ್ರ_8 };

        l_warptile_mmqid_int = { 128,                       128, 128, 32, mul_mat_mm_warp_8 * 2, 64, 2, 4, 4, 1, mul_mat_mm_warp_8 };
        m_warptile_mmqid_int = { 128,                        64,  64, 32, mul_mat_mm_warp_8,     32, 2, 2, 2, 1, mul_mat_mm_warp_8 };
        s_warptile_mmqid_int = { mul_mat_subgroup_ಗಾತ್ರ_32,   32,  32, 32, s_warptile_wm,         32, 2, 2, 1, 1, mul_mat_subgroup_ಗಾತ್ರ_8 };

        l_warptile_mmqid_int_k = { 128,                     128, 128, 32, mul_mat_mm_warp_16 * 2, 64, 1, 4, 4, 1, mul_mat_mm_warp_16 };
        m_warptile_mmqid_int_k = { 128,                      64,  64, 32, mul_mat_mm_warp_16,     32, 1, 2, 2, 1, mul_mat_mm_warp_16 };
        s_warptile_mmqid_int_k = { mul_mat_subgroup_ಗಾತ್ರ_32, 32,  32, 32, s_warptile_wm,          32, 1, 2, 1, 1, mul_mat_subgroup_ಗಾತ್ರ_16 };

        // chip specific tuning
        ಆದರೆ ((device->architecture == AMD_GCN) && (device->driver_id != vk::DriverId::eAmdProprietary)) {
            m_warptile_mmq = m_warptile_mmq_int = { 256, 64, 64, 32, 16, 16, 2, 2, 2, 1, 16 };
            m_warptile_mmqid = m_warptile_mmqid_int = { 256, 64, 64, 32, 16, 16, 2, 2, 2, 1, 16 };
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (device->vendor_id == VK_VENDOR_ID_AMD && device->coopmat_support && device->driver_id != vk::DriverId::eAmdProprietary) {
            // This is intentionally using tx_m values, slight performance increase
            l_warptile = { 256, 128, 128, 16, mm_warp_8, 64, 2, tm_m, tn_m, tk_m, mm_warp_8 };
            l_warptile_mmq = l_warptile_mmq_int = { 256, 128, 128, 32, mm_warp_8, 64, 2, tm_m, tn_m, tk_m, mm_warp_8 };
            l_warptile_mmq_int_k = { 256, 128, 128, 32, mm_warp_16, 64, 1, 4, 2, 1, mm_warp_16 };
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (device->vendor_id == VK_VENDOR_ID_QUALCOMM && device->coopmat_support) {
            m_warptile     = { 64, 64, 64, 16, 64, 64, 1, tm_l, tn_l, tk_l, 64 };
            m_warptile_mmq = { 64, 64, 64, 32, 64, 64, 1, tm_m, tn_m, tk_m, 64 };
        }

        l_mmq_wg_denoms = l_wg_denoms = {128, 128, 1 };
        m_mmq_wg_denoms = m_wg_denoms = { 64,  64, 1 };
        s_mmq_wg_denoms = s_wg_denoms = { 32,  32, 1 };
        l_align = 128;
        m_align =  64;
        s_align =  32;

        ಆದರೆ (device->vendor_id == VK_VENDOR_ID_INTEL && device->coopmat_support) {
            // Xe1/Xe2/Xe3 with coopmat enabled - warptile performance tuning
            l_warptile = { 512, 128, 128, 16, mm_warp_8, 32, 2, tm_l, tn_l, tk_l, mm_warp_8 };
            ಆದರೆ (device->architecture == INTEL_XE1) {
                l_warptile_mmq  = { 512, 256, 128, 32, 32, 32, 2, tm_l, tn_l, tk_l, 16 };
                l_mmq_wg_denoms = { 256, 128, 1 };
                l_align         = 32;  //set as BK
            } ಇಲ್ಲದಿದ್ದರೆ {
                l_warptile_mmq  = { 512, 128, 256, 32, 32, 32, 2, tm_l, tn_l, tk_l, 16 };
                l_mmq_wg_denoms = { 128, 256, 1 };
                l_align         = 32;  //set as BK
            }
        }

        ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ use_cm1_int = device->coopmat_int_support &&
                                 (device->architecture == AMD_RDNA3 || device->architecture == AMD_RDNA4);

        ಗಾಗಿ (uint32_t i = 0; i < GGML_TYPE_ಎಣಿಕೆ; ++i) {
            ggml_type t = (ggml_type)i;
            // Disable medium and large matrix multiplication if not enough shared memory is available
            // Check mmq warptiles as the largest configuration
            // Throw an error if not enough for any matrix multiplication is available
            ಆದರೆ (!ggml_vk_matmul_shmem_support(device, s_warptile_mmq, ಸುಳ್ಳು, t)) {
                std::cerr << "ggml_vulkan: Error: Shared memory size too small for matrix multiplication." << std::ಸಾಲು_ಅಂತ್ಯ;
                ಎಸೆ std::runtime_error("Shared memory size too small for matrix multiplication.");
            } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (!ggml_vk_matmul_shmem_support(device, m_warptile_mmq, ಸುಳ್ಳು, t)) {
                device->mul_mat_m[i] = ಸುಳ್ಳು;
                device->mul_mat_l[i] = ಸುಳ್ಳು;
            } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (!ggml_vk_matmul_shmem_support(device, l_warptile_mmq, ಸುಳ್ಳು, t)) {
                device->mul_mat_l[i] = ಸುಳ್ಳು;
            }

            // Disable mul_mat_id if not enough shared memory is available
            ಆದರೆ (!ggml_vk_matmul_shmem_support(device, s_warptile_mmqid, ನಿಜ, t)) {
                device->mul_mat_id_s[i] = ಸುಳ್ಳು;
                device->mul_mat_id_m[i] = ಸುಳ್ಳು;
                device->mul_mat_id_l[i] = ಸುಳ್ಳು;
            } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (!ggml_vk_matmul_shmem_support(device, m_warptile_mmqid, ನಿಜ, t)) {
                device->mul_mat_id_m[i] = ಸುಳ್ಳು;
                device->mul_mat_id_l[i] = ಸುಳ್ಳು;
            } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (!ggml_vk_matmul_shmem_support(device, l_warptile_mmqid, ನಿಜ, t)) {
                device->mul_mat_id_l[i] = ಸುಳ್ಳು;
            }

            // The q8_1 mmq path has its own (larger) shmem layout, check it separately.
            // K-quants and IQ3_S use the _int_k warptiles, others use _int.
            // cm1 splits k-tiles on the KSCALES==2 types and shares tiles between dense/id.
            ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ is_k_quant = (t == GGML_TYPE_Q2_K || t == GGML_TYPE_Q3_K ||
                                     t == GGML_TYPE_Q4_K || t == GGML_TYPE_Q5_K ||
                                     t == GGML_TYPE_Q6_K || t == GGML_TYPE_IQ3_S);
            ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ cm1_k_tile = (t == GGML_TYPE_Q3_K || t == GGML_TYPE_Q6_K ||
                                     t == GGML_TYPE_NVFP4);

            ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ & s_int   = use_cm1_int ? (cm1_k_tile ? s_warptile_mmq_cm1_int_k : s_warptile_mmq_cm1_int)
                                               : (is_k_quant  ? s_warptile_mmq_int_k     : s_warptile_mmq_int);
            ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ & m_int   = use_cm1_int ? (cm1_k_tile ? m_warptile_mmq_cm1_int_k : m_warptile_mmq_cm1_int)
                                               : (is_k_quant  ? m_warptile_mmq_int_k     : m_warptile_mmq_int);
            ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ & l_int   = use_cm1_int ? (cm1_k_tile ? l_warptile_mmq_cm1_int_k : l_warptile_mmq_cm1_int)
                                               : (is_k_quant  ? l_warptile_mmq_int_k     : l_warptile_mmq_int);
            ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ & s_intid = use_cm1_int ? (cm1_k_tile ? s_warptile_mmq_cm1_int_k : s_warptile_mmq_cm1_int)
                                               : (is_k_quant  ? s_warptile_mmqid_int_k   : s_warptile_mmqid_int);
            ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ & m_intid = use_cm1_int ? (cm1_k_tile ? m_warptile_mmq_cm1_int_k : m_warptile_mmq_cm1_int)
                                               : (is_k_quant  ? m_warptile_mmqid_int_k   : m_warptile_mmqid_int);
            ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ & l_intid = use_cm1_int ? (cm1_k_tile ? l_warptile_mmq_cm1_int_k : l_warptile_mmq_cm1_int)
                                               : (is_k_quant  ? l_warptile_mmqid_int_k   : l_warptile_mmqid_int);

            ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ int_shmem_support = [&](ಸ್ಥಿರಾಂಕ std::vector<uint32_t>& wt, ತಾರ್ಕಿಕ id) {
                ಹಿಂತಿರುಗಿ use_cm1_int ? ggml_vk_matmul_cm1_int_shmem_support(device, wt, id, t)
                                   : ggml_vk_matmul_int_shmem_support(device, wt, id, t);
            };

            ಆದರೆ (!int_shmem_support(s_int, ಸುಳ್ಳು)) {
                device->mul_mat_s_int[i] = ಸುಳ್ಳು;
                device->mul_mat_m_int[i] = ಸುಳ್ಳು;
                device->mul_mat_l_int[i] = ಸುಳ್ಳು;
            } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (!int_shmem_support(m_int, ಸುಳ್ಳು)) {
                device->mul_mat_m_int[i] = ಸುಳ್ಳು;
                device->mul_mat_l_int[i] = ಸುಳ್ಳು;
            } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (!int_shmem_support(l_int, ಸುಳ್ಳು)) {
                device->mul_mat_l_int[i] = ಸುಳ್ಳು;
            }

            ಆದರೆ (!int_shmem_support(s_intid, ನಿಜ)) {
                device->mul_mat_id_s_int[i] = ಸುಳ್ಳು;
                device->mul_mat_id_m_int[i] = ಸುಳ್ಳು;
                device->mul_mat_id_l_int[i] = ಸುಳ್ಳು;
            } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (!int_shmem_support(m_intid, ನಿಜ)) {
                device->mul_mat_id_m_int[i] = ಸುಳ್ಳು;
                device->mul_mat_id_l_int[i] = ಸುಳ್ಳು;
            } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (!int_shmem_support(l_intid, ನಿಜ)) {
                device->mul_mat_id_l_int[i] = ಸುಳ್ಳು;
            }
        }
    }

    ಸ್ವಯಂಚಾಲಿತ ಸ್ಥಿರಾಂಕ &ggml_vk_create_pipeline = [&](vk_device& device, vk_pipeline& ತಳ_pipeline, ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *name, size_t spv_ಗಾತ್ರ, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ* spv_ದತ್ತಾಂಶ, ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *entrypoint,
                                              uint32_t parameter_ಎಣಿಕೆ, uint32_t ತಳ್ಳು_constant_ಗಾತ್ರ, std::array<uint32_t, 3> wg_denoms, ಸ್ಥಿರಾಂಕ std::vector<uint32_t>& specialization_constants,
                                              uint32_t align, ತಾರ್ಕಿಕ disable_robustness = ಸುಳ್ಳು, ತಾರ್ಕಿಕ require_ತುಂಬಿದ_subgroups = ಸುಳ್ಳು, uint32_t required_subgroup_ಗಾತ್ರ = 0) {

        ಆದರೆ (!require_ತುಂಬಿದ_subgroups && required_subgroup_ಗಾತ್ರ == 0) {
            required_subgroup_ಗಾತ್ರ = get_subgroup_ಗಾತ್ರ(name, device->architecture);
        }

        vk_pipeline *ptr = &ತಳ_pipeline;

        ಪೂರ್ಣಾಂಕ ಅಂಕೆ_pipelines = 1;
#ಆದರೆ defined(VK_EXT_shader_64bit_indexing)
        ಆದರೆ (device->shader_64b_indexing) {
            ಅಂಕೆ_pipelines = 2;
        }
#ಆದರೆ_ಅಂತ್ಯ
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < ಅಂಕೆ_pipelines; ++i, ptr = &(*ptr)->ಮುಂದಿನ) {
            vk_pipeline &pipeline = *ptr;
            ಆದರೆ (!pipeline) {
                pipeline = std::make_shared<vk_pipeline_struct>();
            }
            ಆದರೆ (!pipeline->initialized) {
                pipeline->name = name;
                pipeline->parameter_ಎಣಿಕೆ = parameter_ಎಣಿಕೆ;
                pipeline->ತಳ್ಳು_constant_ಗಾತ್ರ = ತಳ್ಳು_constant_ಗಾತ್ರ;
                pipeline->wg_denoms = wg_denoms;
                pipeline->align = align;
                pipeline->initialized = ನಿಜ;
#ಆದರೆ defined(VK_EXT_shader_64bit_indexing)
                pipeline->is_64b_indexing = (i == 1);
#ಆದರೆ_ಅಂತ್ಯ
            }

            // We only care about the pipeline this call asked for; the rest
            // (including the 64-bit indexing variant) are handled by their
            // own request_descriptor_sets / load_shaders calls.
            ಆದರೆ (pipeline.get() != requested.get()) {
                ಮುಂದುವರಿ;
            }

            ಆದರೆ (pipeline->compiled) {
                ಮುಂದುವರಿ;
            }

            wait_pipeline = pipeline;

            ಆದರೆ (!pipeline->compile_pending) {
                pipeline->compile_pending = ನಿಜ;
                claimed_task.pipeline = pipeline;
                claimed_task.spv_ಗಾತ್ರ = spv_ಗಾತ್ರ;
                claimed_task.spv_ದತ್ತಾಂಶ = spv_ದತ್ತಾಂಶ;
                claimed_task.entrypoint = entrypoint;
                claimed_task.parameter_ಎಣಿಕೆ = parameter_ಎಣಿಕೆ;
                claimed_task.wg_denoms = wg_denoms;
                claimed_task.specialization_constants = specialization_constants;
                claimed_task.disable_robustness = disable_robustness;
                claimed_task.require_ತುಂಬಿದ_subgroups = require_ತುಂಬಿದ_subgroups;
                claimed_task.required_subgroup_ಗಾತ್ರ = required_subgroup_ಗಾತ್ರ;
                has_claimed_task = ನಿಜ;
            }
        }
    };

    ಸ್ವಯಂಚಾಲಿತ ಸ್ಥಿರಾಂಕ &ggml_vk_create_pipeline2 = [&](vk_device& device, vk_pipeline& pipeline, ಸ್ಥಿರಾಂಕ std::string &name, size_t spv_ಗಾತ್ರ, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ* spv_ದತ್ತಾಂಶ, ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *entrypoint,
                                              uint32_t parameter_ಎಣಿಕೆ, uint32_t ತಳ್ಳು_constant_ಗಾತ್ರ, std::array<uint32_t, 3> wg_denoms, ಸ್ಥಿರಾಂಕ std::vector<uint32_t>& specialization_constants,
                                              uint32_t align, ತಾರ್ಕಿಕ disable_robustness = ಸುಳ್ಳು, ತಾರ್ಕಿಕ require_ತುಂಬಿದ_subgroups = ಸುಳ್ಳು, uint32_t required_subgroup_ಗಾತ್ರ = 0) {
        ಹಿಂತಿರುಗಿ ggml_vk_create_pipeline(device, pipeline, name.c_str(), spv_ಗಾತ್ರ, spv_ದತ್ತಾಂಶ, entrypoint,
                                       parameter_ಎಣಿಕೆ, ತಳ್ಳು_constant_ಗಾತ್ರ, wg_denoms, specialization_constants,
                                       align, disable_robustness, require_ತುಂಬಿದ_subgroups, required_subgroup_ಗಾತ್ರ);
    };

    // FA scalar has two SPIR-V modules (MMQ vs non-MMQ); FA cm1 has one. K/V
    // quant type is selected at runtime via the FaTypeK / FaTypeV spec constants.

    ಗಾಗಿ (ಸ್ವಯಂಚಾಲಿತ &fa : device->pipeline_flash_attn_f32_f16) {
        ಆದರೆ (fa.ಮೊದಲ.path != FA_SCALAR) ಮುಂದುವರಿ;
        ಸ್ಥಿರಾಂಕ uint32_t Br = fa.ಮೊದಲ.Br;
        ಸ್ಥಿರಾಂಕ uint32_t Bc = fa.ಮೊದಲ.Bc;
        ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ aligned = fa.ಮೊದಲ.aligned;
        ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ f32acc = fa.ಮೊದಲ.f32acc;
        ಸ್ಥಿರಾಂಕ uint32_t fa_sgs = fa.ಮೊದಲ.subgroup_ಗಾತ್ರ;
        ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ fa_ds = fa.ಮೊದಲ.subgroup_ಗಾತ್ರ == 0;

        ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ bf16_kv = fa.ಮೊದಲ.k_type == GGML_TYPE_BF16;
        ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ use_mmq = ggml_vk_fa_scalar_ಬಳಸುತ್ತದೆ_mmq(device, fa.ಮೊದಲ.k_type, fa.ಮೊದಲ.v_type);
        ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * spv_ದತ್ತಾಂಶ = ಶೂನ್ಯಸೂಚಕ;
        size_t spv_ಗಾತ್ರ = 0;
        ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *name = ಶೂನ್ಯಸೂಚಕ;
        ಆದರೆ (bf16_kv) {
            spv_ದತ್ತಾಂಶ = flash_attn_f32_f16_fp32_ದತ್ತಾಂಶ;
            spv_ಗಾತ್ರ = flash_attn_f32_f16_fp32_len;
            name = aligned ? "flash_attn_f32_bf16_aligned" : "flash_attn_f32_bf16";
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (use_mmq) {
#ಆದರೆ defined(GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT)
            ಆದರೆ (device->fp16) {
                ಆದರೆ (f32acc) { spv_ದತ್ತಾಂಶ = flash_attn_f32_f16_int8_ದತ್ತಾಂಶ;        spv_ಗಾತ್ರ = flash_attn_f32_f16_int8_len; }
                ಇಲ್ಲದಿದ್ದರೆ        { spv_ದತ್ತಾಂಶ = flash_attn_f32_f16_f16acc_int8_ದತ್ತಾಂಶ; spv_ಗಾತ್ರ = flash_attn_f32_f16_f16acc_int8_len; }
            } ಇಲ್ಲದಿದ್ದರೆ {
                spv_ದತ್ತಾಂಶ = flash_attn_f32_f16_fp32_int8_ದತ್ತಾಂಶ;
                spv_ಗಾತ್ರ = flash_attn_f32_f16_fp32_int8_len;
            }
#ಆದರೆ_ಅಂತ್ಯ
            name = aligned ? "flash_attn_f32_f16_aligned" : "flash_attn_f32_f16";
        } ಇಲ್ಲದಿದ್ದರೆ {
            ಆದರೆ (device->fp16) {
                ಆದರೆ (device->dot2_f16) {
                    ಆದರೆ (f32acc) { spv_ದತ್ತಾಂಶ = flash_attn_f32_f16_dot2_ದತ್ತಾಂಶ;        spv_ಗಾತ್ರ = flash_attn_f32_f16_dot2_len; }
                    ಇಲ್ಲದಿದ್ದರೆ        { spv_ದತ್ತಾಂಶ = flash_attn_f32_f16_dot2_f16acc_ದತ್ತಾಂಶ; spv_ಗಾತ್ರ = flash_attn_f32_f16_dot2_f16acc_len; }
                } ಇಲ್ಲದಿದ್ದರೆ {
                    ಆದರೆ (f32acc) { spv_ದತ್ತಾಂಶ = flash_attn_f32_f16_ದತ್ತಾಂಶ;        spv_ಗಾತ್ರ = flash_attn_f32_f16_len; }
                    ಇಲ್ಲದಿದ್ದರೆ        { spv_ದತ್ತಾಂಶ = flash_attn_f32_f16_f16acc_ದತ್ತಾಂಶ; spv_ಗಾತ್ರ = flash_attn_f32_f16_f16acc_len; }
                }
            } ಇಲ್ಲದಿದ್ದರೆ {
                spv_ದತ್ತಾಂಶ = flash_attn_f32_f16_fp32_ದತ್ತಾಂಶ;
                spv_ಗಾತ್ರ = flash_attn_f32_f16_fp32_len;
            }
            name = aligned ? "flash_attn_f32_f16_aligned" : "flash_attn_f32_f16";
        }
        ggml_vk_create_pipeline(device, fa.second, name, spv_ಗಾತ್ರ, spv_ದತ್ತಾಂಶ, "main", 8,
                                ಗಾತ್ರ(vk_flash_attn_ತಳ್ಳು_constants), {Br, 1, 1},
                                get_fa_spec_constants(fa.ಮೊದಲ), aligned ? Bc : 1, ನಿಜ,
                                !fa_ds, !fa_ds ? fa_sgs : 0);
    }

#ಆದರೆ defined(VK_KHR_cooperative_ಮಾತೃಕೆ) && defined(GGML_VULKAN_COOPMAT_GLSLC_SUPPORT)
    ಆದರೆ (device->coopmat1_fa_support) {
        ಗಾಗಿ (ಸ್ವಯಂಚಾಲಿತ &fa : device->pipeline_flash_attn_f32_f16) {
            ಆದರೆ (fa.ಮೊದಲ.path != FA_COOPMAT1) ಮುಂದುವರಿ;
            ಸ್ಥಿರಾಂಕ uint32_t Br = fa.ಮೊದಲ.Br;
            ಸ್ಥಿರಾಂಕ uint32_t Bc = fa.ಮೊದಲ.Bc;
            ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ aligned = fa.ಮೊದಲ.aligned;
            ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ f32acc = fa.ಮೊದಲ.f32acc;
            ಸ್ಥಿರಾಂಕ uint32_t fa_sgs = fa.ಮೊದಲ.subgroup_ಗಾತ್ರ;
            ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ fa_ds = fa.ಮೊದಲ.subgroup_ಗಾತ್ರ == 0;

            ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ bf16_kv = fa.ಮೊದಲ.k_type == GGML_TYPE_BF16;

            ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * spv_ದತ್ತಾಂಶ;
            size_t spv_ಗಾತ್ರ;
            ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *name;
            ಆದರೆ (bf16_kv) {
#ಆದರೆ defined(VK_KHR_shader_bfloat16) && defined(GGML_VULKAN_BFLOAT16_GLSLC_SUPPORT)
                ಆದರೆ (!device->coopmat_bf16_support) ಮುಂದುವರಿ;
                spv_ದತ್ತಾಂಶ = flash_attn_f32_f16_bf16_cm1_ದತ್ತಾಂಶ;
                spv_ಗಾತ್ರ = flash_attn_f32_f16_bf16_cm1_len;
                name = aligned ? "flash_attn_f32_bf16_aligned_cm1" : "flash_attn_f32_bf16_cm1";
#ಇಲ್ಲದಿದ್ದರೆ
                ಮುಂದುವರಿ;
#ಆದರೆ_ಅಂತ್ಯ
            } ಇಲ್ಲದಿದ್ದರೆ {
                ಆದರೆ (f32acc) { spv_ದತ್ತಾಂಶ = flash_attn_f32_f16_cm1_ದತ್ತಾಂಶ;        spv_ಗಾತ್ರ = flash_attn_f32_f16_cm1_len; }
                ಇಲ್ಲದಿದ್ದರೆ        { spv_ದತ್ತಾಂಶ = flash_attn_f32_f16_f16acc_cm1_ದತ್ತಾಂಶ; spv_ಗಾತ್ರ = flash_attn_f32_f16_f16acc_cm1_len; }
                name = aligned ? "flash_attn_f32_f16_aligned_cm1" : "flash_attn_f32_f16_cm1";
            }
            ggml_vk_create_pipeline(device, fa.second, name, spv_ಗಾತ್ರ, spv_ದತ್ತಾಂಶ, "main", 8,
                                    ಗಾತ್ರ(vk_flash_attn_ತಳ್ಳು_constants), {Br, 1, 1},
                                    get_fa_spec_constants(fa.ಮೊದಲ), aligned ? Bc : 1, ನಿಜ,
                                    !fa_ds, !fa_ds ? fa_sgs : 0);
        }
    }
#ಆದರೆ_ಅಂತ್ಯ

#ಆದರೆ defined(VK_NV_cooperative_matrix2) && defined(GGML_VULKAN_COOPMAT2_GLSLC_SUPPORT)
    ಆದರೆ (device->coopmat2) {
        ಗಾಗಿ (ಸ್ವಯಂಚಾಲಿತ &fa : device->pipeline_flash_attn_f32_f16) {
            ಆದರೆ (fa.ಮೊದಲ.path != FA_COOPMAT2) ಮುಂದುವರಿ;
            ಸ್ಥಿರಾಂಕ uint32_t Br = fa.ಮೊದಲ.Br;
            ಸ್ಥಿರಾಂಕ uint32_t Bc = fa.ಮೊದಲ.Bc;
            ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ aligned = fa.ಮೊದಲ.aligned;
            ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ f32acc = fa.ಮೊದಲ.f32acc;

            ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ bf16_kv = fa.ಮೊದಲ.k_type == GGML_TYPE_BF16;
            ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * spv_ದತ್ತಾಂಶ;
            size_t spv_ಗಾತ್ರ;
            ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * name;
            ಆದರೆ (bf16_kv) {
#ಆದರೆ defined(VK_KHR_shader_bfloat16) && defined(GGML_VULKAN_BFLOAT16_GLSLC_SUPPORT)
                ಆದರೆ (!device->coopmat2_bf16_support) ಮುಂದುವರಿ;
                spv_ದತ್ತಾಂಶ = flash_attn_f32_f16_bf16_cm2_ದತ್ತಾಂಶ;
                spv_ಗಾತ್ರ = flash_attn_f32_f16_bf16_cm2_len;
                name = aligned ? "flash_attn_f32_bf16_aligned_cm2" : "flash_attn_f32_bf16_cm2";
#ಇಲ್ಲದಿದ್ದರೆ
                ಮುಂದುವರಿ;
#ಆದರೆ_ಅಂತ್ಯ
            } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (aligned) {
                ಆದರೆ (f32acc) { spv_ದತ್ತಾಂಶ = flash_attn_f32_f16_cm2_ದತ್ತಾಂಶ;        spv_ಗಾತ್ರ = flash_attn_f32_f16_cm2_len;        name = "flash_attn_f32_f16_aligned_f32acc_cm2"; }
                ಇಲ್ಲದಿದ್ದರೆ        { spv_ದತ್ತಾಂಶ = flash_attn_f32_f16_f16acc_cm2_ದತ್ತಾಂಶ; spv_ಗಾತ್ರ = flash_attn_f32_f16_f16acc_cm2_len; name = "flash_attn_f32_f16_aligned_f16acc_cm2"; }
            } ಇಲ್ಲದಿದ್ದರೆ {
                ಆದರೆ (f32acc) { spv_ದತ್ತಾಂಶ = flash_attn_f32_f16_cm2_ದತ್ತಾಂಶ;        spv_ಗಾತ್ರ = flash_attn_f32_f16_cm2_len;        name = "flash_attn_f32_f16_f32acc_cm2"; }
                ಇಲ್ಲದಿದ್ದರೆ        { spv_ದತ್ತಾಂಶ = flash_attn_f32_f16_f16acc_cm2_ದತ್ತಾಂಶ; spv_ಗಾತ್ರ = flash_attn_f32_f16_f16acc_cm2_len; name = "flash_attn_f32_f16_f16acc_cm2"; }
            }
            ggml_vk_create_pipeline(device, fa.second, name, spv_ಗಾತ್ರ, spv_ದತ್ತಾಂಶ, "main", 8,
                                    ಗಾತ್ರ(vk_flash_attn_ತಳ್ಳು_constants), {Br, 1, 1},
                                    get_fa_spec_constants(fa.ಮೊದಲ), aligned ? Bc : 1, ನಿಜ, ಸುಳ್ಳು, 0);
        }
    }
#ಆದರೆ_ಅಂತ್ಯ

    ಸ್ವಯಂಚಾಲಿತ ಸ್ಥಿರಾಂಕ &ggml_vk_mul_mm_spec = [&device](std::vector<uint32_t> spec, ತಾರ್ಕಿಕ aligned) {
        spec.push_back(aligned ? 1u : 0u);  // constantID=11: ALIGNED
        ಆದರೆ (device->vendor_id == VK_VENDOR_ID_INTEL && device->coopmat_support &&
            device->driver_id == vk::DriverId::eIntelProprietaryWindows) {
            spec.push_back(0u);  // constantID=12: SHMEM_STRIDE_PAD = 0
            spec.push_back(1u);  // constantID=13: APPLY_SLM_A_RESHAPE = true
        }
        ಹಿಂತಿರುಗಿ spec;
    };

    ಸ್ವಯಂಚಾಲಿತ ಸ್ಥಿರಾಂಕ &ggml_vk_mul_mm_spec_quant = [&device](std::vector<uint32_t> spec, ತಾರ್ಕಿಕ aligned, uint32_t type) {
        spec.push_back(aligned ? 1u : 0u);  // constantID=11: ALIGNED
        spec.push_back(type);               // constantID=12: MmTypeA
        ಆದರೆ (device->vendor_id == VK_VENDOR_ID_INTEL && device->coopmat_support &&
            device->driver_id == vk::DriverId::eIntelProprietaryWindows) {
            spec.push_back(0u);  // constantID=13: SHMEM_STRIDE_PAD = 0
            spec.push_back(1u);  // constantID=14: APPLY_SLM_A_RESHAPE = true
        }
        ಹಿಂತಿರುಗಿ spec;
    };

    ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ ggml_type non_lut_quant_types[] = {
        GGML_TYPE_Q1_0, GGML_TYPE_Q2_0, GGML_TYPE_Q4_0, GGML_TYPE_Q4_1, GGML_TYPE_Q5_0, GGML_TYPE_Q5_1, GGML_TYPE_Q8_0,
        GGML_TYPE_Q2_K, GGML_TYPE_Q3_K, GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K, GGML_TYPE_TQ1_0, GGML_TYPE_TQ2_0,
    };

#ವ್ಯಾಖ್ಯಾನಿಸು FOR_EACH_LUT_TYPE_NONFP4(X) \
    X(GGML_TYPE_IQ1_S,   iq1_s)   \
    X(GGML_TYPE_IQ1_M,   iq1_m)   \
    X(GGML_TYPE_IQ2_XXS, iq2_xxs) \
    X(GGML_TYPE_IQ2_XS,  iq2_xs)  \
    X(GGML_TYPE_IQ2_S,   iq2_s)   \
    X(GGML_TYPE_IQ3_XXS, iq3_xxs) \
    X(GGML_TYPE_IQ3_S,   iq3_s)   \
    X(GGML_TYPE_IQ4_XS,  iq4_xs)  \
    X(GGML_TYPE_IQ4_NL,  iq4_nl)
#ವ್ಯಾಖ್ಯಾನಿಸು FOR_EACH_LUT_FP4_TYPE(X) \
    X(GGML_TYPE_MXFP4,   mxfp4)   \
    X(GGML_TYPE_NVFP4,   nvfp4)
#ವ್ಯಾಖ್ಯಾನಿಸು FOR_EACH_LUT_TYPE(X) \
    FOR_EACH_LUT_TYPE_NONFP4(X)  \
    FOR_EACH_LUT_FP4_TYPE(X)

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ mul_mat_id_param_ಎಣಿಕೆ = 5;

    ಬಳಸಿ spec_fn_t = std::function<std::vector<uint32_t>(ಸ್ಥಿರಾಂಕ std::vector<uint32_t>&, ತಾರ್ಕಿಕ)>;
    ಸ್ವಯಂಚಾಲಿತ ಸ್ಥಿರಾಂಕ &create_mm_pipelines = [&](
        ಸ್ಥಿರಾಂಕ vk_matmul_pipeline_ಕೀಲಿ& ಕೀಲಿ,
        ಸ್ಥಿರಾಂಕ std::vector<vk_tile_config>& tile_configs,
        ಸ್ಥಿರಾಂಕ std::string& shader_name, size_t spv_len, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ* spv_ದತ್ತಾಂಶ,
        uint32_t ತಳ್ಳು_constant_ಗಾತ್ರ, uint32_t param_ಎಣಿಕೆ,
        ಸ್ಥಿರಾಂಕ spec_fn_t& spec_fn,
        ತಾರ್ಕಿಕ disable_robustness = ಸುಳ್ಳು, ತಾರ್ಕಿಕ require_ತುಂಬಿದ_subgroups = ಸುಳ್ಳು, uint32_t required_subgroup_ಗಾತ್ರ = 0,
        ತಾರ್ಕಿಕ create_aligned = ನಿಜ, ತಾರ್ಕಿಕ pin_subgroup_ಗಮ್ಯ_warp = ಸುಳ್ಳು
    ) {
        ಸ್ವಯಂಚಾಲಿತ& vec = device->pipeline_matmul[ಕೀಲಿ];
        ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ ಮೊದಲ_call = vec.ಖಾಲಿಗಳು();
        ಗಾಗಿ (size_t i = 0; i < tile_configs.size(); i++) {
            ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ& tc = tile_configs[i];

            // Intel coopmat1 pins the required subgroup size to each warptile's WARP element.
            ಸ್ಥಿರಾಂಕ uint32_t rsgs = pin_subgroup_ಗಮ್ಯ_warp ? tc.warptile[WARP_ಗಾತ್ರ_IDX] : required_subgroup_ಗಾತ್ರ;
            ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ     rfs  = require_ತುಂಬಿದ_subgroups || pin_subgroup_ಗಮ್ಯ_warp;

            ಆದರೆ (ಮೊದಲ_call) {
                vk_matmul_pipeline_pair pair{};
                pair.align = tc.align;
                std::string suffix = "_" + std::to_string(i);
                pair.unaligned = std::make_shared<vk_pipeline_struct>();
                ಆದರೆ (create_aligned) {
                    pair.aligned = std::make_shared<vk_pipeline_struct>();
                }
                vec.push_back(pair);
            }

            ggml_vk_create_pipeline(device, vec[i].unaligned,
                vec[i].unaligned->name.ಖಾಲಿಗಳು() ? (shader_name + "_" + std::to_string(i)).c_str() : vec[i].unaligned->name.c_str(),
                spv_len, spv_ದತ್ತಾಂಶ, "main", param_ಎಣಿಕೆ, ತಳ್ಳು_constant_ಗಾತ್ರ,
                tc.wg_denoms, spec_fn(tc.warptile, ಸುಳ್ಳು), 1,
                disable_robustness, rfs, rsgs);

            ಆದರೆ (vec[i].aligned) {
                ggml_vk_create_pipeline(device, vec[i].aligned,
                    vec[i].aligned->name.ಖಾಲಿಗಳು() ? (shader_name + "_aligned_" + std::to_string(i)).c_str() : vec[i].aligned->name.c_str(),
                    spv_len, spv_ದತ್ತಾಂಶ, "main", param_ಎಣಿಕೆ, ತಳ್ಳು_constant_ಗಾತ್ರ,
                    tc.wg_denoms, spec_fn(tc.warptile, ನಿಜ), tc.align,
                    disable_robustness, rfs, rsgs);
            }
        }
    };

    ಸ್ವಯಂಚಾಲಿತ filter_tc = [&](ಸ್ಥಿರಾಂಕ std::vector<vk_tile_config>& configs, ggml_type type, ತಾರ್ಕಿಕ is_id, ತಾರ್ಕಿಕ is_int = ಸುಳ್ಳು) -> std::vector<vk_tile_config> {
        std::vector<vk_tile_config> ಫಲಿತಾಂಶ;
        ತಾರ್ಕಿಕ enabled[3];
        ಆದರೆ (is_int) {
            enabled[0] = is_id ? device->mul_mat_id_s_int[type] : device->mul_mat_s_int[type];
            enabled[1] = is_id ? device->mul_mat_id_m_int[type] : device->mul_mat_m_int[type];
            enabled[2] = is_id ? device->mul_mat_id_l_int[type] : device->mul_mat_l_int[type];
        } ಇಲ್ಲದಿದ್ದರೆ {
            enabled[0] = is_id ? device->mul_mat_id_s[type] : device->mul_mat_s[type];
            enabled[1] = is_id ? device->mul_mat_id_m[type] : device->mul_mat_m[type];
            enabled[2] = is_id ? device->mul_mat_id_l[type] : device->mul_mat_l[type];
        }
        ಗಾಗಿ (size_t i = 0; i < configs.size() && i < 3; i++) {
            ಆದರೆ (enabled[i]) ಫಲಿತಾಂಶ.push_back(configs[i]);
        }
        ಹಿಂತಿರುಗಿ ಫಲಿತಾಂಶ;
    };

    std::vector<vk_tile_config> tc_mm = {{s_warptile, s_wg_denoms, s_align}, {m_warptile, m_wg_denoms, m_align}, {l_warptile, l_wg_denoms, l_align}};
    std::vector<vk_tile_config> tc_mmq = {{s_warptile_mmq, s_mmq_wg_denoms, s_align}, {m_warptile_mmq, m_mmq_wg_denoms, m_align}, {l_warptile_mmq, l_mmq_wg_denoms, l_align}};

#ಆದರೆ defined(VK_NV_cooperative_matrix2) && defined(GGML_VULKAN_COOPMAT2_GLSLC_SUPPORT)
    ಆದರೆ (device->coopmat2) {
        ಸ್ವಯಂಚಾಲಿತ ಸ್ಥಿರಾಂಕ &ggml_vk_mul_mm_cm2_spec = [&](std::vector<uint32_t> spec, ತಾರ್ಕಿಕ aligned, uint32_t type = UINT32_MAX) {
            spec.push_back(aligned ? 1u : 0u);        // ALIGNED
            spec.push_back(device->subgroup_ಗಾತ್ರ);     // subgroup_size
            ಆದರೆ (type != UINT32_MAX) {
                spec.push_back(type);                  // MmTypeA
                spec.push_back((uint32_t)ggml_type_ಗಾತ್ರ((ggml_type)type)); // MmABlockBytes
            }
            ಹಿಂತಿರುಗಿ spec;
        };

        std::vector<vk_tile_config> tc_mmq_k = {{s_warptile_mmq_k, s_mmq_wg_denoms_k, s_align}, {m_warptile_mmq_k, m_mmq_wg_denoms_k, m_align}, {l_warptile_mmq_k, l_mmq_wg_denoms_k, l_align}};
        std::vector<vk_tile_config> tc_mmqid = {{s_warptile_mmqid, s_mmqid_wg_denoms, s_align}, {m_warptile_mmqid, m_mmqid_wg_denoms, m_align}, {l_warptile_mmqid, l_mmqid_wg_denoms, l_align}};

        spec_fn_t cm2_spec = [&](ಸ್ಥಿರಾಂಕ std::vector<uint32_t>& wt, ತಾರ್ಕಿಕ a) { ಹಿಂತಿರುಗಿ ggml_vk_mul_mm_cm2_spec(wt, a); };

        // F16 x F16
        create_mm_pipelines({GGML_TYPE_F16, GGML_TYPE_F16, ಸುಳ್ಳು, ನಿಜ},  tc_mm, "matmul_f16_f16acc", matmul_f16_f16acc_cm2_len, matmul_f16_f16acc_cm2_ದತ್ತಾಂಶ, ಗಾತ್ರ(vk_mat_mat_ತಳ್ಳು_constants), 3, cm2_spec, ನಿಜ);
        create_mm_pipelines({GGML_TYPE_F16, GGML_TYPE_F16, ಸುಳ್ಳು, ಸುಳ್ಳು}, tc_mm, "matmul_f16",        matmul_f16_cm2_len,        matmul_f16_cm2_ದತ್ತಾಂಶ,        ಗಾತ್ರ(vk_mat_mat_ತಳ್ಳು_constants), 3, cm2_spec, ನಿಜ);
#ಆದರೆ defined(GGML_VULKAN_BFLOAT16_GLSLC_SUPPORT)
        ಆದರೆ (device->coopmat_bf16_support) {
            create_mm_pipelines({GGML_TYPE_BF16, GGML_TYPE_BF16, ಸುಳ್ಳು, ಸುಳ್ಳು}, tc_mm, "matmul_bf16", matmul_bf16_cm2_len, matmul_bf16_cm2_ದತ್ತಾಂಶ, ಗಾತ್ರ(vk_mat_mat_ತಳ್ಳು_constants), 3, cm2_spec, ನಿಜ);
        }
#ಆದರೆ_ಅಂತ್ಯ
        ಗಾಗಿ (ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ type : non_lut_quant_types) {
            // regression in unified shader on Ampere
            ಆದರೆ (type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K) {
                ಮುಂದುವರಿ;
            }
            ಸ್ವಯಂಚಾಲಿತ& tc = ((type >= GGML_TYPE_Q2_K && type <= GGML_TYPE_Q6_K) || type == GGML_TYPE_TQ1_0 || type == GGML_TYPE_TQ2_0) ? tc_mmq_k : tc_mmq;
            spec_fn_t qs = [&, type](ಸ್ಥಿರಾಂಕ std::vector<uint32_t>& wt, ತಾರ್ಕಿಕ a) { ಹಿಂತಿರುಗಿ ggml_vk_mul_mm_cm2_spec(wt, a, (uint32_t)type); };
            create_mm_pipelines({type, GGML_TYPE_F16, ಸುಳ್ಳು, ನಿಜ},  tc, "matmul_quant_f16_f16acc", matmul_quant_f16_f16acc_cm2_len, matmul_quant_f16_f16acc_cm2_ದತ್ತಾಂಶ, ಗಾತ್ರ(vk_mat_mat_ತಳ್ಳು_constants), 3, qs, ನಿಜ);
            create_mm_pipelines({type, GGML_TYPE_F16, ಸುಳ್ಳು, ಸುಳ್ಳು}, tc, "matmul_quant_f16",        matmul_quant_f16_cm2_len,        matmul_quant_f16_cm2_ದತ್ತಾಂಶ,        ಗಾತ್ರ(vk_mat_mat_ತಳ್ಳು_constants), 3, qs, ನಿಜ);
        }
        create_mm_pipelines({GGML_TYPE_Q4_K, GGML_TYPE_F16, ಸುಳ್ಳು, ನಿಜ},  tc_mmq_k, "matmul_q4_k_f16_f16acc", matmul_q4_k_f16_f16acc_cm2_len, matmul_q4_k_f16_f16acc_cm2_ದತ್ತಾಂಶ, ಗಾತ್ರ(vk_mat_mat_ತಳ್ಳು_constants), 3, cm2_spec, ನಿಜ);
        create_mm_pipelines({GGML_TYPE_Q4_K, GGML_TYPE_F16, ಸುಳ್ಳು, ಸುಳ್ಳು}, tc_mmq_k, "matmul_q4_k_f16",        matmul_q4_k_f16_cm2_len,        matmul_q4_k_f16_cm2_ದತ್ತಾಂಶ,        ಗಾತ್ರ(vk_mat_mat_ತಳ್ಳು_constants), 3, cm2_spec, ನಿಜ);
        create_mm_pipelines({GGML_TYPE_Q5_K, GGML_TYPE_F16, ಸುಳ್ಳು, ನಿಜ},  tc_mmq_k, "matmul_q5_k_f16_f16acc", matmul_q5_k_f16_f16acc_cm2_len, matmul_q5_k_f16_f16acc_cm2_ದತ್ತಾಂಶ, ಗಾತ್ರ(vk_mat_mat_ತಳ್ಳು_constants), 3, cm2_spec, ನಿಜ);
        create_mm_pipelines({GGML_TYPE_Q5_K, GGML_TYPE_F16, ಸುಳ್ಳು, ಸುಳ್ಳು}, tc_mmq_k, "matmul_q5_k_f16",        matmul_q5_k_f16_cm2_len,        matmul_q5_k_f16_cm2_ದತ್ತಾಂಶ,        ಗಾತ್ರ(vk_mat_mat_ತಳ್ಳು_constants), 3, cm2_spec, ನಿಜ);
#ವ್ಯಾಖ್ಯಾನಿಸು X_CM2(TYPE, tstr) \
        { ಸ್ವಯಂಚಾಲಿತ tc = filter_tc(tc_mmq, TYPE, ಸುಳ್ಳು); \
          ಆದರೆ (!tc.ಖಾಲಿಗಳು()) { \
              create_mm_pipelines({TYPE, GGML_TYPE_F16, ಸುಳ್ಳು, ನಿಜ},  tc, "matmul_" #tstr "_f16_f16acc", matmul_##tstr##_f16_f16acc_cm2_len, matmul_##tstr##_f16_f16acc_cm2_data, ಗಾತ್ರ(vk_mat_mat_ತಳ್ಳು_constants), 3, cm2_spec, ನಿಜ); \
              create_mm_pipelines({TYPE, GGML_TYPE_F16, ಸುಳ್ಳು, ಸುಳ್ಳು}, tc, "matmul_" #tstr "_f16",        matmul_##tstr##_f16_cm2_len,        matmul_##tstr##_f16_cm2_data,        ಗಾತ್ರ(vk_mat_mat_ತಳ್ಳು_constants), 3, cm2_spec, ನಿಜ); \
          } }
        FOR_EACH_LUT_TYPE_NONFP4(X_CM2)
#ಆದರೆ defined(GGML_VULKAN_FLOAT_E2M1_GLSLC_SUPPORT) && defined(GGML_VULKAN_FLOAT_E4M3_GLSLC_SUPPORT)
        ಆದರೆ (device->ocp_fp4) {
#ವ್ಯಾಖ್ಯಾನಿಸು X_CM2_OCP(TYPE, tstr) \
            { ಸ್ವಯಂಚಾಲಿತ tc = filter_tc(tc_mmq, TYPE, ಸುಳ್ಳು); \
              ಆದರೆ (!tc.ಖಾಲಿಗಳು()) { \
                  create_mm_pipelines({TYPE, GGML_TYPE_F16, ಸುಳ್ಳು, ನಿಜ},  tc, "matmul_" #tstr "_f16_ocp_f16acc", matmul_##tstr##_f16_ocp_f16acc_cm2_len, matmul_##tstr##_f16_ocp_f16acc_cm2_data, ಗಾತ್ರ(vk_mat_mat_ತಳ್ಳು_constants), 3, cm2_spec, ನಿಜ); \
                  create_mm_pipelines({TYPE, GGML_TYPE_F16, ಸುಳ್ಳು, ಸುಳ್ಳು}, tc, "matmul_" #tstr "_f16_ocp",        matmul_##tstr##_f16_ocp_cm2_len,        matmul_##tstr##_f16_ocp_cm2_data,        ಗಾತ್ರ(vk_mat_mat_ತಳ್ಳು_constants), 3, cm2_spec, ನಿಜ); \
              } }
            FOR_EACH_LUT_FP4_TYPE(X_CM2_OCP)
#ವ್ಯಾಖ್ಯಾನ_ರದ್ದು X_CM2_OCP
        } ಇಲ್ಲದಿದ್ದರೆ
#ಆದರೆ_ಅಂತ್ಯ
        {
            FOR_EACH_LUT_FP4_TYPE(X_CM2)
        }
#ವ್ಯಾಖ್ಯಾನ_ರದ್ದು X_CM2

        GGML_ASSERT(device->subgroup_ballot);

        create_mm_pipelines({GGML_TYPE_F16, GGML_TYPE_F16, ನಿಜ, ನಿಜ},  tc_mm, "matmul_id_subgroup_f16_f16acc", matmul_id_subgroup_f16_f16acc_cm2_len, matmul_id_subgroup_f16_f16acc_cm2_ದತ್ತಾಂಶ, ಗಾತ್ರ(vk_mat_mat_id_ತಳ್ಳು_constants), 5, cm2_spec, ನಿಜ);
        create_mm_pipelines({GGML_TYPE_F16, GGML_TYPE_F16, ನಿಜ, ಸುಳ್ಳು}, tc_mm, "matmul_id_subgroup_f16",        matmul_id_subgroup_f16_cm2_len,        matmul_id_subgroup_f16_cm2_ದತ್ತಾಂಶ,        ಗಾತ್ರ(vk_mat_mat_id_ತಳ್ಳು_constants), 5, cm2_spec, ನಿಜ);
#ಆದರೆ defined(GGML_VULKAN_BFLOAT16_GLSLC_SUPPORT)
        ಆದರೆ (device->coopmat_bf16_support) {
            create_mm_pipelines({GGML_TYPE_BF16, GGML_TYPE_BF16, ನಿಜ, ಸುಳ್ಳು}, tc_mm, "matmul_id_subgroup_bf16", matmul_id_subgroup_bf16_cm2_len, matmul_id_subgroup_bf16_cm2_ದತ್ತಾಂಶ, ಗಾತ್ರ(vk_mat_mat_id_ತಳ್ಳು_constants), 5, cm2_spec, ನಿಜ);
        }
#ಆದರೆ_ಅಂತ್ಯ
        ಗಾಗಿ (ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ type : non_lut_quant_types) {
            ಆದರೆ (type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K) {
                ಮುಂದುವರಿ;
            }
            spec_fn_t qs_id = [&, type](ಸ್ಥಿರಾಂಕ std::vector<uint32_t>& wt, ತಾರ್ಕಿಕ a) { ಹಿಂತಿರುಗಿ ggml_vk_mul_mm_cm2_spec(wt, a, (uint32_t)type); };
            create_mm_pipelines({type, GGML_TYPE_F16, ನಿಜ, ನಿಜ},  tc_mmqid, "matmul_id_subgroup_quant_f16_f16acc", matmul_id_subgroup_quant_f16_f16acc_cm2_len, matmul_id_subgroup_quant_f16_f16acc_cm2_ದತ್ತಾಂಶ, ಗಾತ್ರ(vk_mat_mat_id_ತಳ್ಳು_constants), 5, qs_id, ನಿಜ);
            create_mm_pipelines({type, GGML_TYPE_F16, ನಿಜ, ಸುಳ್ಳು}, tc_mmqid, "matmul_id_subgroup_quant_f16",        matmul_id_subgroup_quant_f16_cm2_len,        matmul_id_subgroup_quant_f16_cm2_ದತ್ತಾಂಶ,        ಗಾತ್ರ(vk_mat_mat_id_ತಳ್ಳು_constants), 5, qs_id, ನಿಜ);
        }
        create_mm_pipelines({GGML_TYPE_Q4_K, GGML_TYPE_F16, ನಿಜ, ನಿಜ},  tc_mmqid, "matmul_id_subgroup_q4_k_f16_f16acc", matmul_id_subgroup_q4_k_f16_f16acc_cm2_len, matmul_id_subgroup_q4_k_f16_f16acc_cm2_ದತ್ತಾಂಶ, ಗಾತ್ರ(vk_mat_mat_id_ತಳ್ಳು_constants), 5, cm2_spec, ನಿಜ);
        create_mm_pipelines({GGML_TYPE_Q4_K, GGML_TYPE_F16, ನಿಜ, ಸುಳ್ಳು}, tc_mmqid, "matmul_id_subgroup_q4_k_f16",        matmul_id_subgroup_q4_k_f16_cm2_len,        matmul_id_subgroup_q4_k_f16_cm2_ದತ್ತಾಂಶ,        ಗಾತ್ರ(vk_mat_mat_id_ತಳ್ಳು_constants), 5, cm2_spec, ನಿಜ);
        create_mm_pipelines({GGML_TYPE_Q5_K, GGML_TYPE_F16, ನಿಜ, ನಿಜ},  tc_mmqid, "matmul_id_subgroup_q5_k_f16_f16acc", matmul_id_subgroup_q5_k_f16_f16acc_cm2_len, matmul_id_subgroup_q5_k_f16_f16acc_cm2_ದತ್ತಾಂಶ, ಗಾತ್ರ(vk_mat_mat_id_ತಳ್ಳು_constants), 5, cm2_spec, ನಿಜ);
        create_mm_pipelines({GGML_TYPE_Q5_K, GGML_TYPE_F16, ನಿಜ, ಸುಳ್ಳು}, tc_mmqid, "matmul_id_subgroup_q5_k_f16",        matmul_id_subgroup_q5_k_f16_cm2_len,        matmul_id_subgroup_q5_k_f16_cm2_ದತ್ತಾಂಶ,        ಗಾತ್ರ(vk_mat_mat_id_ತಳ್ಳು_constants), 5, cm2_spec, ನಿಜ);
#ವ್ಯಾಖ್ಯಾನಿಸು X_CM2_ID(TYPE, tstr) \
        { ಸ್ವಯಂಚಾಲಿತ tc = filter_tc(tc_mmqid, TYPE, ನಿಜ); \
          ಆದರೆ (!tc.ಖಾಲಿಗಳು()) { \
              create_mm_pipelines({TYPE, GGML_TYPE_F16, ನಿಜ, ನಿಜ},  tc, "matmul_id_subgroup_" #tstr "_f16_f16acc", matmul_id_subgroup_##tstr##_f16_f16acc_cm2_len, matmul_id_subgroup_##tstr##_f16_f16acc_cm2_data, ಗಾತ್ರ(vk_mat_mat_id_ತಳ್ಳು_constants), 5, cm2_spec, ನಿಜ); \
              create_mm_pipelines({TYPE, GGML_TYPE_F16, ನಿಜ, ಸುಳ್ಳು}, tc, "matmul_id_subgroup_" #tstr "_f16",        matmul_id_subgroup_##tstr##_f16_cm2_len,        matmul_id_subgroup_##tstr##_f16_cm2_data,        ಗಾತ್ರ(vk_mat_mat_id_ತಳ್ಳು_constants), 5, cm2_spec, ನಿಜ); \
          } }
        FOR_EACH_LUT_TYPE_NONFP4(X_CM2_ID)
#ಆದರೆ defined(GGML_VULKAN_FLOAT_E2M1_GLSLC_SUPPORT) && defined(GGML_VULKAN_FLOAT_E4M3_GLSLC_SUPPORT)
        ಆದರೆ (device->ocp_fp4) {
#ವ್ಯಾಖ್ಯಾನಿಸು X_CM2_ID_OCP(TYPE, tstr) \
            { ಸ್ವಯಂಚಾಲಿತ tc = filter_tc(tc_mmqid, TYPE, ನಿಜ); \
              ಆದರೆ (!tc.ಖಾಲಿಗಳು()) { \
                  create_mm_pipelines({TYPE, GGML_TYPE_F16, ನಿಜ, ನಿಜ},  tc, "matmul_id_subgroup_" #tstr "_f16_ocp_f16acc", matmul_id_subgroup_##tstr##_f16_ocp_f16acc_cm2_len, matmul_id_subgroup_##tstr##_f16_ocp_f16acc_cm2_data, ಗಾತ್ರ(vk_mat_mat_id_ತಳ್ಳು_constants), 5, cm2_spec, ನಿಜ); \
                  create_mm_pipelines({TYPE, GGML_TYPE_F16, ನಿಜ, ಸುಳ್ಳು}, tc, "matmul_id_subgroup_" #tstr "_f16_ocp",        matmul_id_subgroup_##tstr##_f16_ocp_cm2_len,        matmul_id_subgroup_##tstr##_f16_ocp_cm2_data,        ಗಾತ್ರ(vk_mat_mat_id_ತಳ್ಳು_constants), 5, cm2_spec, ನಿಜ); \
              } }
            FOR_EACH_LUT_FP4_TYPE(X_CM2_ID_OCP)
#ವ್ಯಾಖ್ಯಾನ_ರದ್ದು X_CM2_ID_OCP
        } ಇಲ್ಲದಿದ್ದರೆ
#ಆದರೆ_ಅಂತ್ಯ
        {
            FOR_EACH_LUT_FP4_TYPE(X_CM2_ID)
        }
#ವ್ಯಾಖ್ಯಾನ_ರದ್ದು X_CM2_ID
    } ಇಲ್ಲದಿದ್ದರೆ
#ಆದರೆ_ಅಂತ್ಯ  // defined(VK_NV_cooperative_matrix2) && defined(GGML_VULKAN_COOPMAT2_GLSLC_SUPPORT)
#ಆದರೆ defined(VK_KHR_cooperative_ಮಾತೃಕೆ) && defined(GGML_VULKAN_COOPMAT_GLSLC_SUPPORT)
    ಆದರೆ (device->coopmat_support) {
        spec_fn_t cm1_spec = [&](ಸ್ಥಿರಾಂಕ std::vector<uint32_t>& wt, ತಾರ್ಕಿಕ a) { ಹಿಂತಿರುಗಿ ggml_vk_mul_mm_spec(wt, a); };

        // Intel coopmat1 pins each pipeline's required subgroup size to its warptile WARP element.
        ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ cm1_pin = device->vendor_id == VK_VENDOR_ID_INTEL;

        // Intel coopmat1 uses a dedicated large-tile config for quant matmul_id.
        std::vector<vk_tile_config> tc_mmq_id = tc_mmq;
        ಆದರೆ (cm1_pin) {
            tc_mmq_id[2] = { { 512, 128, 128, 32, 32, 32, 2, device->coopmat_m, device->coopmat_n, device->coopmat_k, 32 }, { 128, 128, 1 }, 32 };
        }

        ಸ್ವಯಂಚಾಲಿತ cm1_create = [&](vk_matmul_pipeline_ಕೀಲಿ ಕೀಲಿ, ಸ್ಥಿರಾಂಕ std::vector<vk_tile_config>& tc_ತಳ,
                              ಸ್ಥಿರಾಂಕ std::string& name, size_t len, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ* ದತ್ತಾಂಶ, uint32_t pc_ಗಾತ್ರ, uint32_t pc) {
            ಸ್ವಯಂಚಾಲಿತ tc = filter_tc(tc_ತಳ, ಕೀಲಿ.type_a, ಕೀಲಿ.mul_mat_id);
            ಆದರೆ (!tc.ಖಾಲಿಗಳು()) create_mm_pipelines(ಕೀಲಿ, tc, name, len, ದತ್ತಾಂಶ, pc_ಗಾತ್ರ, pc, cm1_spec, ಸುಳ್ಳು, ನಿಜ, 0, ನಿಜ, cm1_pin);
        };
        ಸ್ವಯಂಚಾಲಿತ cm1_create_quant = [&](vk_matmul_pipeline_ಕೀಲಿ ಕೀಲಿ, ಸ್ಥಿರಾಂಕ std::vector<vk_tile_config>& tc_ತಳ,
                                    ಸ್ಥಿರಾಂಕ std::string& name, size_t len, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ* ದತ್ತಾಂಶ, uint32_t pc_ಗಾತ್ರ, uint32_t pc) {
            spec_fn_t qs = [&, type_a=ಕೀಲಿ.type_a](ಸ್ಥಿರಾಂಕ std::vector<uint32_t>& wt, ತಾರ್ಕಿಕ a) { ಹಿಂತಿರುಗಿ ggml_vk_mul_mm_spec_quant(wt, a, (uint32_t)type_a); };
            ಸ್ವಯಂಚಾಲಿತ tc = filter_tc(tc_ತಳ, ಕೀಲಿ.type_a, ಕೀಲಿ.mul_mat_id);
            ಆದರೆ (!tc.ಖಾಲಿಗಳು()) create_mm_pipelines(ಕೀಲಿ, tc, name, len, ದತ್ತಾಂಶ, pc_ಗಾತ್ರ, pc, qs, ಸುಳ್ಳು, ನಿಜ, 0, ನಿಜ, cm1_pin);
        };
        // int8 MMQ helper: per-type cm1 shader, warptile passed as-is (carries DEVICE_ARCH in
        // spec constant WARP_SIZE_IDX+1), subgroup size pinned to the warptile WARP element, no aligned variant.
        ಸ್ವಯಂಚಾಲಿತ cm1_create_mmq = [&](vk_matmul_pipeline_ಕೀಲಿ ಕೀಲಿ, ಸ್ಥಿರಾಂಕ std::vector<vk_tile_config>& tc_ತಳ,
                                  ಸ್ಥಿರಾಂಕ std::string& name, size_t len, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ* ದತ್ತಾಂಶ, uint32_t pc_ಗಾತ್ರ, uint32_t pc) {
            spec_fn_t ತಟಸ್ಥ = [](ಸ್ಥಿರಾಂಕ std::vector<uint32_t>& wt, ತಾರ್ಕಿಕ) { ಹಿಂತಿರುಗಿ wt; };
            ಸ್ವಯಂಚಾಲಿತ tc = filter_tc(tc_ತಳ, ಕೀಲಿ.type_a, ಕೀಲಿ.mul_mat_id, ನಿಜ);
            ಆದರೆ (!tc.ಖಾಲಿಗಳು()) create_mm_pipelines(ಕೀಲಿ, tc, name, len, ದತ್ತಾಂಶ, pc_ಗಾತ್ರ, pc, ತಟಸ್ಥ, ಸುಳ್ಳು, ಸುಳ್ಳು, 0, ಸುಳ್ಳು, ನಿಜ);
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
        ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ rdna3 = device->architecture == AMD_RDNA3;
        ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ rdna4 = device->architecture == AMD_RDNA4;

        cm1_create({GGML_TYPE_F32, GGML_TYPE_F32, ಸುಳ್ಳು, ಸುಳ್ಳು}, tc_mm, "matmul_f32_f32",     matmul_f32_f32_cm1_len,     matmul_f32_f32_cm1_ದತ್ತಾಂಶ,     ಗಾತ್ರ(vk_mat_mat_ತಳ್ಳು_constants), 3);
        cm1_create({GGML_TYPE_F32, GGML_TYPE_F16, ಸುಳ್ಳು, ಸುಳ್ಳು}, tc_mm, "matmul_f32_f16",     matmul_f32_f16_cm1_len,     matmul_f32_f16_cm1_ದತ್ತಾಂಶ,     ಗಾತ್ರ(vk_mat_mat_ತಳ್ಳು_constants), 3);
        ಆದರೆ (device->coopmat_acc_f16_support) {
            cm1_create({GGML_TYPE_F16, GGML_TYPE_F16, ಸುಳ್ಳು, ನಿಜ},  tc_mm, "matmul_f16_f16acc", matmul_f16_f16acc_cm1_len, matmul_f16_f16acc_cm1_ದತ್ತಾಂಶ, ಗಾತ್ರ(vk_mat_mat_ತಳ್ಳು_constants), 3);
            cm1_create({GGML_TYPE_F16, GGML_TYPE_F32, ಸುಳ್ಳು, ನಿಜ},  tc_mm, "matmul_f16_f32_f16acc", matmul_f16_f32_f16acc_cm1_len, matmul_f16_f32_f16acc_cm1_ದತ್ತಾಂಶ, ಗಾತ್ರ(vk_mat_mat_ತಳ್ಳು_constants), 3);
        }
        ಆದರೆ (device->coopmat_acc_f32_support) {
            cm1_create({GGML_TYPE_F16, GGML_TYPE_F16, ಸುಳ್ಳು, ಸುಳ್ಳು}, tc_mm, "matmul_f16",      matmul_f16_cm1_len,      matmul_f16_cm1_ದತ್ತಾಂಶ,      ಗಾತ್ರ(vk_mat_mat_ತಳ್ಳು_constants), 3);
            cm1_create({GGML_TYPE_F16, GGML_TYPE_F32, ಸುಳ್ಳು, ಸುಳ್ಳು}, tc_mm, "matmul_f16_f32",  matmul_f16_f32_cm1_len,  matmul_f16_f32_cm1_ದತ್ತಾಂಶ,  ಗಾತ್ರ(vk_mat_mat_ತಳ್ಳು_constants), 3);
        }
#ಆದರೆ defined(GGML_VULKAN_BFLOAT16_GLSLC_SUPPORT)
        ಆದರೆ (device->coopmat_bf16_support) {
            cm1_create({GGML_TYPE_BF16, GGML_TYPE_BF16, ಸುಳ್ಳು, ಸುಳ್ಳು}, tc_mm, "matmul_bf16", matmul_bf16_cm1_len, matmul_bf16_cm1_ದತ್ತಾಂಶ, ಗಾತ್ರ(vk_mat_mat_ತಳ್ಳು_constants), 3);
        }
#ಆದರೆ_ಅಂತ್ಯ
        ಗಾಗಿ (ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ type : non_lut_quant_types) {
            ಆದರೆ (device->coopmat_acc_f16_support) {
                cm1_create_quant({type, GGML_TYPE_F32, ಸುಳ್ಳು, ನಿಜ},  tc_mmq, "matmul_quant_f32_f16acc", matmul_quant_f32_f16acc_cm1_len, matmul_quant_f32_f16acc_cm1_ದತ್ತಾಂಶ, ಗಾತ್ರ(vk_mat_mat_ತಳ್ಳು_constants), 3);
                cm1_create_quant({type, GGML_TYPE_F16, ಸುಳ್ಳು, ನಿಜ},  tc_mmq, "matmul_quant_f16_f16acc", matmul_quant_f16_f16acc_cm1_len, matmul_quant_f16_f16acc_cm1_ದತ್ತಾಂಶ, ಗಾತ್ರ(vk_mat_mat_ತಳ್ಳು_constants), 3);
            }
            ಆದರೆ (device->coopmat_acc_f32_support) {
                cm1_create_quant({type, GGML_TYPE_F32, ಸುಳ್ಳು, ಸುಳ್ಳು}, tc_mmq, "matmul_quant_f32",        matmul_quant_f32_cm1_len,        matmul_quant_f32_cm1_ದತ್ತಾಂಶ,        ಗಾತ್ರ(vk_mat_mat_ತಳ್ಳು_constants), 3);
                cm1_create_quant({type, GGML_TYPE_F16, ಸುಳ್ಳು, ಸುಳ್ಳು}, tc_mmq, "matmul_quant_f16",        matmul_quant_f16_cm1_len,        matmul_quant_f16_cm1_ದತ್ತಾಂಶ,        ಗಾತ್ರ(vk_mat_mat_ತಳ್ಳು_constants), 3);
            }
        }
        // The _f16 variants provide the f16 B-type pipeline used when y_non_contig converts f32->f16.
#ವ್ಯಾಖ್ಯಾನಿಸು X_CM1(TYPE, tstr) \
        ಆದರೆ (device->coopmat_acc_f16_support) { \
            cm1_create({TYPE, GGML_TYPE_F32, ಸುಳ್ಳು, ನಿಜ},  tc_mmq, "matmul_" #tstr "_f32_f16acc", matmul_##tstr##_f32_f16acc_cm1_len, matmul_##tstr##_f32_f16acc_cm1_data, ಗಾತ್ರ(vk_mat_mat_ತಳ್ಳು_constants), 3); \
            cm1_create({TYPE, GGML_TYPE_F16, ಸುಳ್ಳು, ನಿಜ},  tc_mmq, "matmul_" #tstr "_f16_f16acc", matmul_##tstr##_f16_f16acc_cm1_len, matmul_##tstr##_f16_f16acc_cm1_data, ಗಾತ್ರ(vk_mat_mat_ತಳ್ಳು_constants), 3); \
        } \
        ಆದರೆ (device->coopmat_acc_f32_support) { \
            cm1_create({TYPE, GGML_TYPE_F32, ಸುಳ್ಳು, ಸುಳ್ಳು}, tc_mmq, "matmul_" #tstr "_f32",        matmul_##tstr##_f32_cm1_len,        matmul_##tstr##_f32_cm1_data,        ಗಾತ್ರ(vk_mat_mat_ತಳ್ಳು_constants), 3); \
            cm1_create({TYPE, GGML_TYPE_F16, ಸುಳ್ಳು, ಸುಳ್ಳು}, tc_mmq, "matmul_" #tstr "_f16",        matmul_##tstr##_f16_cm1_len,        matmul_##tstr##_f16_cm1_data,        ಗಾತ್ರ(vk_mat_mat_ತಳ್ಳು_constants), 3); \
        }
        FOR_EACH_LUT_TYPE_NONFP4(X_CM1)
#ಆದರೆ defined(GGML_VULKAN_FLOAT_E2M1_GLSLC_SUPPORT) && defined(GGML_VULKAN_FLOAT_E4M3_GLSLC_SUPPORT)
        ಆದರೆ (device->ocp_fp4) {
#ವ್ಯಾಖ್ಯಾನಿಸು X_CM1_OCP(TYPE, tstr) \
            ಆದರೆ (device->coopmat_acc_f16_support) { \
                cm1_create({TYPE, GGML_TYPE_F32, ಸುಳ್ಳು, ನಿಜ},  tc_mmq, "matmul_" #tstr "_f32_ocp_f16acc", matmul_##tstr##_f32_ocp_f16acc_cm1_len, matmul_##tstr##_f32_ocp_f16acc_cm1_data, ಗಾತ್ರ(vk_mat_mat_ತಳ್ಳು_constants), 3); \
                cm1_create({TYPE, GGML_TYPE_F16, ಸುಳ್ಳು, ನಿಜ},  tc_mmq, "matmul_" #tstr "_f16_ocp_f16acc", matmul_##tstr##_f16_ocp_f16acc_cm1_len, matmul_##tstr##_f16_ocp_f16acc_cm1_data, ಗಾತ್ರ(vk_mat_mat_ತಳ್ಳು_constants), 3); \
            } \
            ಆದರೆ (device->coopmat_acc_f32_support) { \
                cm1_create({TYPE, GGML_TYPE_F32, ಸುಳ್ಳು, ಸುಳ್ಳು}, tc_mmq, "matmul_" #tstr "_f32_ocp",        matmul_##tstr##_f32_ocp_cm1_len,        matmul_##tstr##_f32_ocp_cm1_data,        ಗಾತ್ರ(vk_mat_mat_ತಳ್ಳು_constants), 3); \
                cm1_create({TYPE, GGML_TYPE_F16, ಸುಳ್ಳು, ಸುಳ್ಳು}, tc_mmq, "matmul_" #tstr "_f16_ocp",        matmul_##tstr##_f16_ocp_cm1_len,        matmul_##tstr##_f16_ocp_cm1_data,        ಗಾತ್ರ(vk_mat_mat_ತಳ್ಳು_constants), 3); \
            }
            FOR_EACH_LUT_FP4_TYPE(X_CM1_OCP)
#ವ್ಯಾಖ್ಯಾನ_ರದ್ದು X_CM1_OCP
        } ಇಲ್ಲದಿದ್ದರೆ
#ಆದರೆ_ಅಂತ್ಯ
        {
            FOR_EACH_LUT_FP4_TYPE(X_CM1)
        }
#ವ್ಯಾಖ್ಯಾನ_ರದ್ದು X_CM1

        ಆದರೆ (device->coopmat_int_support && (rdna3 || rdna4)) {
            cm1_create_mmq({GGML_TYPE_Q4_0,   GGML_TYPE_Q8_1, ಸುಳ್ಳು, ಸುಳ್ಳು}, tc_mmq_cm1_int,   "matmul_q4_0_q8_1",   matmul_q4_0_q8_1_cm1_len,   matmul_q4_0_q8_1_cm1_ದತ್ತಾಂಶ,   ಗಾತ್ರ(vk_mat_mat_ತಳ್ಳು_constants), 3);
            ಆದರೆ (!rdna4) { cm1_create_mmq({GGML_TYPE_Q4_1, GGML_TYPE_Q8_1, ಸುಳ್ಳು, ಸುಳ್ಳು}, tc_mmq_cm1_int,   "matmul_q4_1_q8_1",   matmul_q4_1_q8_1_cm1_len,   matmul_q4_1_q8_1_cm1_ದತ್ತಾಂಶ,   ಗಾತ್ರ(vk_mat_mat_ತಳ್ಳು_constants), 3); }
            cm1_create_mmq({GGML_TYPE_Q5_0,   GGML_TYPE_Q8_1, ಸುಳ್ಳು, ಸುಳ್ಳು}, tc_mmq_cm1_int,   "matmul_q5_0_q8_1",   matmul_q5_0_q8_1_cm1_len,   matmul_q5_0_q8_1_cm1_ದತ್ತಾಂಶ,   ಗಾತ್ರ(vk_mat_mat_ತಳ್ಳು_constants), 3);
            ಆದರೆ (!rdna4) { cm1_create_mmq({GGML_TYPE_Q5_1, GGML_TYPE_Q8_1, ಸುಳ್ಳು, ಸುಳ್ಳು}, tc_mmq_cm1_int,   "matmul_q5_1_q8_1",   matmul_q5_1_q8_1_cm1_len,   matmul_q5_1_q8_1_cm1_ದತ್ತಾಂಶ,   ಗಾತ್ರ(vk_mat_mat_ತಳ್ಳು_constants), 3); }
            cm1_create_mmq({GGML_TYPE_Q8_0,   GGML_TYPE_Q8_1, ಸುಳ್ಳು, ಸುಳ್ಳು}, tc_mmq_cm1_int,   "matmul_q8_0_q8_1",   matmul_q8_0_q8_1_cm1_len,   matmul_q8_0_q8_1_cm1_ದತ್ತಾಂಶ,   ಗಾತ್ರ(vk_mat_mat_ತಳ್ಳು_constants), 3);
            cm1_create_mmq({GGML_TYPE_IQ4_NL, GGML_TYPE_Q8_1, ಸುಳ್ಳು, ಸುಳ್ಳು}, tc_mmq_cm1_int,   "matmul_iq4_nl_q8_1", matmul_iq4_nl_q8_1_cm1_len, matmul_iq4_nl_q8_1_cm1_ದತ್ತಾಂಶ, ಗಾತ್ರ(vk_mat_mat_ತಳ್ಳು_constants), 3);
            cm1_create_mmq({GGML_TYPE_IQ4_XS, GGML_TYPE_Q8_1, ಸುಳ್ಳು, ಸುಳ್ಳು}, tc_mmq_cm1_int,   "matmul_iq4_xs_q8_1", matmul_iq4_xs_q8_1_cm1_len, matmul_iq4_xs_q8_1_cm1_ದತ್ತಾಂಶ, ಗಾತ್ರ(vk_mat_mat_ತಳ್ಳು_constants), 3);
            cm1_create_mmq({GGML_TYPE_MXFP4,  GGML_TYPE_Q8_1, ಸುಳ್ಳು, ಸುಳ್ಳು}, tc_mmq_cm1_int,   "matmul_mxfp4_q8_1",  matmul_mxfp4_q8_1_cm1_len,  matmul_mxfp4_q8_1_cm1_ದತ್ತಾಂಶ,  ಗಾತ್ರ(vk_mat_mat_ತಳ್ಳು_constants), 3);
            cm1_create_mmq({GGML_TYPE_Q3_K,   GGML_TYPE_Q8_1, ಸುಳ್ಳು, ಸುಳ್ಳು}, tc_mmq_cm1_int_k, "matmul_q3_k_q8_1",   matmul_q3_k_q8_1_cm1_len,   matmul_q3_k_q8_1_cm1_ದತ್ತಾಂಶ,   ಗಾತ್ರ(vk_mat_mat_ತಳ್ಳು_constants), 3);
            ಆದರೆ (!rdna4) { cm1_create_mmq({GGML_TYPE_Q4_K, GGML_TYPE_Q8_1, ಸುಳ್ಳು, ಸುಳ್ಳು}, tc_mmq_cm1_int,   "matmul_q4_k_q8_1",   matmul_q4_k_q8_1_cm1_len,   matmul_q4_k_q8_1_cm1_ದತ್ತಾಂಶ,   ಗಾತ್ರ(vk_mat_mat_ತಳ್ಳು_constants), 3); }
            ಆದರೆ (!rdna4) { cm1_create_mmq({GGML_TYPE_Q5_K, GGML_TYPE_Q8_1, ಸುಳ್ಳು, ಸುಳ್ಳು}, tc_mmq_cm1_int,   "matmul_q5_k_q8_1",   matmul_q5_k_q8_1_cm1_len,   matmul_q5_k_q8_1_cm1_ದತ್ತಾಂಶ,   ಗಾತ್ರ(vk_mat_mat_ತಳ್ಳು_constants), 3); }
            cm1_create_mmq({GGML_TYPE_Q6_K,   GGML_TYPE_Q8_1, ಸುಳ್ಳು, ಸುಳ್ಳು}, tc_mmq_cm1_int_k, "matmul_q6_k_q8_1",   matmul_q6_k_q8_1_cm1_len,   matmul_q6_k_q8_1_cm1_ದತ್ತಾಂಶ,   ಗಾತ್ರ(vk_mat_mat_ತಳ್ಳು_constants), 3);
            ಆದರೆ (!rdna4) { cm1_create_mmq({GGML_TYPE_NVFP4, GGML_TYPE_Q8_1, ಸುಳ್ಳು, ಸುಳ್ಳು}, tc_mmq_cm1_int_k, "matmul_nvfp4_q8_1",  matmul_nvfp4_q8_1_cm1_len,  matmul_nvfp4_q8_1_cm1_ದತ್ತಾಂಶ,  ಗಾತ್ರ(vk_mat_mat_ತಳ್ಳು_constants), 3); }
        }

        GGML_ASSERT(device->subgroup_ballot);

        cm1_create({GGML_TYPE_F32, GGML_TYPE_F32, ನಿಜ, ಸುಳ್ಳು}, tc_mm, "matmul_id_subgroup_f32_f32", matmul_id_subgroup_f32_f32_cm1_len, matmul_id_subgroup_f32_f32_cm1_ದತ್ತಾಂಶ, ಗಾತ್ರ(vk_mat_mat_id_ತಳ್ಳು_constants), mul_mat_id_param_ಎಣಿಕೆ);
        ಆದರೆ (device->coopmat_acc_f16_support) {
            cm1_create({GGML_TYPE_F16, GGML_TYPE_F16, ನಿಜ, ನಿಜ},  tc_mm, "matmul_id_subgroup_f16_f16acc",     matmul_id_subgroup_f16_f16acc_cm1_len,     matmul_id_subgroup_f16_f16acc_cm1_ದತ್ತಾಂಶ,     ಗಾತ್ರ(vk_mat_mat_id_ತಳ್ಳು_constants), mul_mat_id_param_ಎಣಿಕೆ);
            cm1_create({GGML_TYPE_F16, GGML_TYPE_F32, ನಿಜ, ನಿಜ},  tc_mm, "matmul_id_subgroup_f16_f32_f16acc", matmul_id_subgroup_f16_f32_f16acc_cm1_len, matmul_id_subgroup_f16_f32_f16acc_cm1_ದತ್ತಾಂಶ, ಗಾತ್ರ(vk_mat_mat_id_ತಳ್ಳು_constants), mul_mat_id_param_ಎಣಿಕೆ);
        }
        ಆದರೆ (device->coopmat_acc_f32_support) {
            cm1_create({GGML_TYPE_F16, GGML_TYPE_F16, ನಿಜ, ಸುಳ್ಳು}, tc_mm, "matmul_id_subgroup_f16",     matmul_id_subgroup_f16_cm1_len,     matmul_id_subgroup_f16_cm1_ದತ್ತಾಂಶ,     ಗಾತ್ರ(vk_mat_mat_id_ತಳ್ಳು_constants), mul_mat_id_param_ಎಣಿಕೆ);
            cm1_create({GGML_TYPE_F16, GGML_TYPE_F32, ನಿಜ, ಸುಳ್ಳು}, tc_mm, "matmul_id_subgroup_f16_f32", matmul_id_subgroup_f16_f32_cm1_len, matmul_id_subgroup_f16_f32_cm1_ದತ್ತಾಂಶ, ಗಾತ್ರ(vk_mat_mat_id_ತಳ್ಳು_constants), mul_mat_id_param_ಎಣಿಕೆ);
        }
#ಆದರೆ defined(GGML_VULKAN_BFLOAT16_GLSLC_SUPPORT)
        ಆದರೆ (device->coopmat_bf16_support) {
            cm1_create({GGML_TYPE_BF16, GGML_TYPE_BF16, ನಿಜ, ಸುಳ್ಳು}, tc_mm, "matmul_id_subgroup_bf16", matmul_id_subgroup_bf16_cm1_len, matmul_id_subgroup_bf16_cm1_ದತ್ತಾಂಶ, ಗಾತ್ರ(vk_mat_mat_id_ತಳ್ಳು_constants), mul_mat_id_param_ಎಣಿಕೆ);
        }
#ಆದರೆ_ಅಂತ್ಯ
        ಗಾಗಿ (ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ type : non_lut_quant_types) {
            ಆದರೆ (device->coopmat_acc_f16_support) {
                cm1_create_quant({type, GGML_TYPE_F32, ನಿಜ, ನಿಜ},  tc_mmq_id, "matmul_id_subgroup_quant_f32_f16acc", matmul_id_subgroup_quant_f32_f16acc_cm1_len, matmul_id_subgroup_quant_f32_f16acc_cm1_ದತ್ತಾಂಶ, ಗಾತ್ರ(vk_mat_mat_id_ತಳ್ಳು_constants), mul_mat_id_param_ಎಣಿಕೆ);
                cm1_create_quant({type, GGML_TYPE_F16, ನಿಜ, ನಿಜ},  tc_mmq_id, "matmul_id_subgroup_quant_f16_f16acc", matmul_id_subgroup_quant_f16_f16acc_cm1_len, matmul_id_subgroup_quant_f16_f16acc_cm1_ದತ್ತಾಂಶ, ಗಾತ್ರ(vk_mat_mat_id_ತಳ್ಳು_constants), mul_mat_id_param_ಎಣಿಕೆ);
            }
            ಆದರೆ (device->coopmat_acc_f32_support) {
                cm1_create_quant({type, GGML_TYPE_F32, ನಿಜ, ಸುಳ್ಳು}, tc_mmq_id, "matmul_id_subgroup_quant_f32",        matmul_id_subgroup_quant_f32_cm1_len,        matmul_id_subgroup_quant_f32_cm1_ದತ್ತಾಂಶ,        ಗಾತ್ರ(vk_mat_mat_id_ತಳ್ಳು_constants), mul_mat_id_param_ಎಣಿಕೆ);
                cm1_create_quant({type, GGML_TYPE_F16, ನಿಜ, ಸುಳ್ಳು}, tc_mmq_id, "matmul_id_subgroup_quant_f16",        matmul_id_subgroup_quant_f16_cm1_len,        matmul_id_subgroup_quant_f16_cm1_ದತ್ತಾಂಶ,        ಗಾತ್ರ(vk_mat_mat_id_ತಳ್ಳು_constants), mul_mat_id_param_ಎಣಿಕೆ);
            }
        }
        // The _f16 variants provide the f16 B-type pipeline used when y_non_contig converts f32->f16.
#ವ್ಯಾಖ್ಯಾನಿಸು X_CM1_ID(TYPE, tstr) \
        ಆದರೆ (device->coopmat_acc_f16_support) { \
            cm1_create({TYPE, GGML_TYPE_F32, ನಿಜ, ನಿಜ},  tc_mmq_id, "matmul_id_subgroup_" #tstr "_f32_f16acc", matmul_id_subgroup_##tstr##_f32_f16acc_cm1_len, matmul_id_subgroup_##tstr##_f32_f16acc_cm1_data, ಗಾತ್ರ(vk_mat_mat_id_ತಳ್ಳು_constants), mul_mat_id_param_ಎಣಿಕೆ); \
            cm1_create({TYPE, GGML_TYPE_F16, ನಿಜ, ನಿಜ},  tc_mmq_id, "matmul_id_subgroup_" #tstr "_f16_f16acc", matmul_id_subgroup_##tstr##_f16_f16acc_cm1_len, matmul_id_subgroup_##tstr##_f16_f16acc_cm1_data, ಗಾತ್ರ(vk_mat_mat_id_ತಳ್ಳು_constants), mul_mat_id_param_ಎಣಿಕೆ); \
        } \
        ಆದರೆ (device->coopmat_acc_f32_support) { \
            cm1_create({TYPE, GGML_TYPE_F32, ನಿಜ, ಸುಳ್ಳು}, tc_mmq_id, "matmul_id_subgroup_" #tstr "_f32",        matmul_id_subgroup_##tstr##_f32_cm1_len,        matmul_id_subgroup_##tstr##_f32_cm1_data,        ಗಾತ್ರ(vk_mat_mat_id_ತಳ್ಳು_constants), mul_mat_id_param_ಎಣಿಕೆ); \
            cm1_create({TYPE, GGML_TYPE_F16, ನಿಜ, ಸುಳ್ಳು}, tc_mmq_id, "matmul_id_subgroup_" #tstr "_f16",        matmul_id_subgroup_##tstr##_f16_cm1_len,        matmul_id_subgroup_##tstr##_f16_cm1_data,        ಗಾತ್ರ(vk_mat_mat_id_ತಳ್ಳು_constants), mul_mat_id_param_ಎಣಿಕೆ); \
        }
        FOR_EACH_LUT_TYPE_NONFP4(X_CM1_ID)
#ಆದರೆ defined(GGML_VULKAN_FLOAT_E2M1_GLSLC_SUPPORT) && defined(GGML_VULKAN_FLOAT_E4M3_GLSLC_SUPPORT)
        ಆದರೆ (device->ocp_fp4) {
#ವ್ಯಾಖ್ಯಾನಿಸು X_CM1_ID_OCP(TYPE, tstr) \
            ಆದರೆ (device->coopmat_acc_f16_support) { \
                cm1_create({TYPE, GGML_TYPE_F32, ನಿಜ, ನಿಜ},  tc_mmq_id, "matmul_id_subgroup_" #tstr "_f32_ocp_f16acc", matmul_id_subgroup_##tstr##_f32_ocp_f16acc_cm1_len, matmul_id_subgroup_##tstr##_f32_ocp_f16acc_cm1_data, ಗಾತ್ರ(vk_mat_mat_id_ತಳ್ಳು_constants), mul_mat_id_param_ಎಣಿಕೆ); \
                cm1_create({TYPE, GGML_TYPE_F16, ನಿಜ, ನಿಜ},  tc_mmq_id, "matmul_id_subgroup_" #tstr "_f16_ocp_f16acc", matmul_id_subgroup_##tstr##_f16_ocp_f16acc_cm1_len, matmul_id_subgroup_##tstr##_f16_ocp_f16acc_cm1_data, ಗಾತ್ರ(vk_mat_mat_id_ತಳ್ಳು_constants), mul_mat_id_param_ಎಣಿಕೆ); \
            } \
            ಆದರೆ (device->coopmat_acc_f32_support) { \
                cm1_create({TYPE, GGML_TYPE_F32, ನಿಜ, ಸುಳ್ಳು}, tc_mmq_id, "matmul_id_subgroup_" #tstr "_f32_ocp",        matmul_id_subgroup_##tstr##_f32_ocp_cm1_len,        matmul_id_subgroup_##tstr##_f32_ocp_cm1_data,        ಗಾತ್ರ(vk_mat_mat_id_ತಳ್ಳು_constants), mul_mat_id_param_ಎಣಿಕೆ); \
                cm1_create({TYPE, GGML_TYPE_F16, ನಿಜ, ಸುಳ್ಳು}, tc_mmq_id, "matmul_id_subgroup_" #tstr "_f16_ocp",        matmul_id_subgroup_##tstr##_f16_ocp_cm1_len,        matmul_id_subgroup_##tstr##_f16_ocp_cm1_data,        ಗಾತ್ರ(vk_mat_mat_id_ತಳ್ಳು_constants), mul_mat_id_param_ಎಣಿಕೆ); \
            }
            FOR_EACH_LUT_FP4_TYPE(X_CM1_ID_OCP)
#ವ್ಯಾಖ್ಯಾನ_ರದ್ದು X_CM1_ID_OCP
        } ಇಲ್ಲದಿದ್ದರೆ
#ಆದರೆ_ಅಂತ್ಯ
        {
            FOR_EACH_LUT_FP4_TYPE(X_CM1_ID)
        }
#ವ್ಯಾಖ್ಯಾನ_ರದ್ದು X_CM1_ID

        ಆದರೆ (device->coopmat_int_support && (rdna3 || rdna4)) {
            cm1_create_mmq({GGML_TYPE_Q4_0,   GGML_TYPE_Q8_1, ನಿಜ, ಸುಳ್ಳು}, tc_mmq_cm1_int,   "matmul_id_subgroup_q4_0_q8_1",   matmul_id_subgroup_q4_0_q8_1_cm1_len,   matmul_id_subgroup_q4_0_q8_1_cm1_ದತ್ತಾಂಶ,   ಗಾತ್ರ(vk_mat_mat_id_ತಳ್ಳು_constants), mul_mat_id_param_ಎಣಿಕೆ);
            cm1_create_mmq({GGML_TYPE_Q4_1,   GGML_TYPE_Q8_1, ನಿಜ, ಸುಳ್ಳು}, tc_mmq_cm1_int,   "matmul_id_subgroup_q4_1_q8_1",   matmul_id_subgroup_q4_1_q8_1_cm1_len,   matmul_id_subgroup_q4_1_q8_1_cm1_ದತ್ತಾಂಶ,   ಗಾತ್ರ(vk_mat_mat_id_ತಳ್ಳು_constants), mul_mat_id_param_ಎಣಿಕೆ);
            cm1_create_mmq({GGML_TYPE_Q5_0,   GGML_TYPE_Q8_1, ನಿಜ, ಸುಳ್ಳು}, tc_mmq_cm1_int,   "matmul_id_subgroup_q5_0_q8_1",   matmul_id_subgroup_q5_0_q8_1_cm1_len,   matmul_id_subgroup_q5_0_q8_1_cm1_ದತ್ತಾಂಶ,   ಗಾತ್ರ(vk_mat_mat_id_ತಳ್ಳು_constants), mul_mat_id_param_ಎಣಿಕೆ);
            cm1_create_mmq({GGML_TYPE_Q5_1,   GGML_TYPE_Q8_1, ನಿಜ, ಸುಳ್ಳು}, tc_mmq_cm1_int,   "matmul_id_subgroup_q5_1_q8_1",   matmul_id_subgroup_q5_1_q8_1_cm1_len,   matmul_id_subgroup_q5_1_q8_1_cm1_ದತ್ತಾಂಶ,   ಗಾತ್ರ(vk_mat_mat_id_ತಳ್ಳು_constants), mul_mat_id_param_ಎಣಿಕೆ);
            cm1_create_mmq({GGML_TYPE_Q8_0,   GGML_TYPE_Q8_1, ನಿಜ, ಸುಳ್ಳು}, tc_mmq_cm1_int,   "matmul_id_subgroup_q8_0_q8_1",   matmul_id_subgroup_q8_0_q8_1_cm1_len,   matmul_id_subgroup_q8_0_q8_1_cm1_ದತ್ತಾಂಶ,   ಗಾತ್ರ(vk_mat_mat_id_ತಳ್ಳು_constants), mul_mat_id_param_ಎಣಿಕೆ);
            cm1_create_mmq({GGML_TYPE_IQ4_NL, GGML_TYPE_Q8_1, ನಿಜ, ಸುಳ್ಳು}, tc_mmq_cm1_int,   "matmul_id_subgroup_iq4_nl_q8_1", matmul_id_subgroup_iq4_nl_q8_1_cm1_len, matmul_id_subgroup_iq4_nl_q8_1_cm1_ದತ್ತಾಂಶ, ಗಾತ್ರ(vk_mat_mat_id_ತಳ್ಳು_constants), mul_mat_id_param_ಎಣಿಕೆ);
            cm1_create_mmq({GGML_TYPE_IQ4_XS, GGML_TYPE_Q8_1, ನಿಜ, ಸುಳ್ಳು}, tc_mmq_cm1_int,   "matmul_id_subgroup_iq4_xs_q8_1", matmul_id_subgroup_iq4_xs_q8_1_cm1_len, matmul_id_subgroup_iq4_xs_q8_1_cm1_ದತ್ತಾಂಶ, ಗಾತ್ರ(vk_mat_mat_id_ತಳ್ಳು_constants), mul_mat_id_param_ಎಣಿಕೆ);
            cm1_create_mmq({GGML_TYPE_MXFP4,  GGML_TYPE_Q8_1, ನಿಜ, ಸುಳ್ಳು}, tc_mmq_cm1_int,   "matmul_id_subgroup_mxfp4_q8_1",  matmul_id_subgroup_mxfp4_q8_1_cm1_len,  matmul_id_subgroup_mxfp4_q8_1_cm1_ದತ್ತಾಂಶ,  ಗಾತ್ರ(vk_mat_mat_id_ತಳ್ಳು_constants), mul_mat_id_param_ಎಣಿಕೆ);
            cm1_create_mmq({GGML_TYPE_Q3_K,   GGML_TYPE_Q8_1, ನಿಜ, ಸುಳ್ಳು}, tc_mmq_cm1_int_k, "matmul_id_subgroup_q3_k_q8_1",   matmul_id_subgroup_q3_k_q8_1_cm1_len,   matmul_id_subgroup_q3_k_q8_1_cm1_ದತ್ತಾಂಶ,   ಗಾತ್ರ(vk_mat_mat_id_ತಳ್ಳು_constants), mul_mat_id_param_ಎಣಿಕೆ);
            cm1_create_mmq({GGML_TYPE_Q4_K,   GGML_TYPE_Q8_1, ನಿಜ, ಸುಳ್ಳು}, tc_mmq_cm1_int,   "matmul_id_subgroup_q4_k_q8_1",   matmul_id_subgroup_q4_k_q8_1_cm1_len,   matmul_id_subgroup_q4_k_q8_1_cm1_ದತ್ತಾಂಶ,   ಗಾತ್ರ(vk_mat_mat_id_ತಳ್ಳು_constants), mul_mat_id_param_ಎಣಿಕೆ);
            cm1_create_mmq({GGML_TYPE_Q5_K,   GGML_TYPE_Q8_1, ನಿಜ, ಸುಳ್ಳು}, tc_mmq_cm1_int,   "matmul_id_subgroup_q5_k_q8_1",   matmul_id_subgroup_q5_k_q8_1_cm1_len,   matmul_id_subgroup_q5_k_q8_1_cm1_ದತ್ತಾಂಶ,   ಗಾತ್ರ(vk_mat_mat_id_ತಳ್ಳು_constants), mul_mat_id_param_ಎಣಿಕೆ);
            cm1_create_mmq({GGML_TYPE_Q6_K,   GGML_TYPE_Q8_1, ನಿಜ, ಸುಳ್ಳು}, tc_mmq_cm1_int_k, "matmul_id_subgroup_q6_k_q8_1",   matmul_id_subgroup_q6_k_q8_1_cm1_len,   matmul_id_subgroup_q6_k_q8_1_cm1_ದತ್ತಾಂಶ,   ಗಾತ್ರ(vk_mat_mat_id_ತಳ್ಳು_constants), mul_mat_id_param_ಎಣಿಕೆ);
            ಆದರೆ (!rdna4) { cm1_create_mmq({GGML_TYPE_NVFP4, GGML_TYPE_Q8_1, ನಿಜ, ಸುಳ್ಳು}, tc_mmq_cm1_int_k, "matmul_id_subgroup_nvfp4_q8_1",  matmul_id_subgroup_nvfp4_q8_1_cm1_len,  matmul_id_subgroup_nvfp4_q8_1_cm1_ದತ್ತಾಂಶ,  ಗಾತ್ರ(vk_mat_mat_id_ತಳ್ಳು_constants), mul_mat_id_param_ಎಣಿಕೆ); }
        }
    } ಇಲ್ಲದಿದ್ದರೆ
#ಆದರೆ_ಅಂತ್ಯ  // defined(VK_KHR_cooperative_matrix) && defined(GGML_VULKAN_COOPMAT_GLSLC_SUPPORT)
    {
        // Helper for subgroup path with dot2 selection and filtering
        ಸ್ವಯಂಚಾಲಿತ sg_create = [&](vk_matmul_pipeline_ಕೀಲಿ ಕೀಲಿ, ಸ್ಥಿರಾಂಕ std::vector<vk_tile_config>& tc_ತಳ,
                             ಸ್ಥಿರಾಂಕ std::string& name, size_t len, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ* ದತ್ತಾಂಶ, uint32_t pc_ಗಾತ್ರ, uint32_t pc,
                             uint32_t rsgs = 0) {
            ಸ್ವಯಂಚಾಲಿತ tc = filter_tc(tc_ತಳ, ಕೀಲಿ.type_a, ಕೀಲಿ.mul_mat_id);
            ಆದರೆ (!tc.ಖಾಲಿಗಳು()) create_mm_pipelines(ಕೀಲಿ, tc, name, len, ದತ್ತಾಂಶ, pc_ಗಾತ್ರ, pc,
                [&](ಸ್ಥಿರಾಂಕ std::vector<uint32_t>& wt, ತಾರ್ಕಿಕ a) { ಹಿಂತಿರುಗಿ ggml_vk_mul_mm_spec(wt, a); },
                ಸುಳ್ಳು, rsgs > 0, rsgs);
        };
        ಸ್ವಯಂಚಾಲಿತ sg_create_quant = [&](vk_matmul_pipeline_ಕೀಲಿ ಕೀಲಿ, ಸ್ಥಿರಾಂಕ std::vector<vk_tile_config>& tc_ತಳ,
                                   ಸ್ಥಿರಾಂಕ std::string& name, size_t len, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ* ದತ್ತಾಂಶ, uint32_t pc_ಗಾತ್ರ, uint32_t pc,
                                   uint32_t rsgs = 0) {
            ಸ್ವಯಂಚಾಲಿತ tc = filter_tc(tc_ತಳ, ಕೀಲಿ.type_a, ಕೀಲಿ.mul_mat_id);
            ಆದರೆ (!tc.ಖಾಲಿಗಳು()) {
                spec_fn_t qs = [&, type_a=ಕೀಲಿ.type_a](ಸ್ಥಿರಾಂಕ std::vector<uint32_t>& wt, ತಾರ್ಕಿಕ a) { ಹಿಂತಿರುಗಿ ggml_vk_mul_mm_spec_quant(wt, a, (uint32_t)type_a); };
                create_mm_pipelines(ಕೀಲಿ, tc, name, len, ದತ್ತಾಂಶ, pc_ಗಾತ್ರ, pc, qs, ಸುಳ್ಳು, rsgs > 0, rsgs);
            }
        };
#ಆದರೆ defined(GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT)
        ಸ್ವಯಂಚಾಲಿತ sg_create_mmq = [&](vk_matmul_pipeline_ಕೀಲಿ ಕೀಲಿ, ಸ್ಥಿರಾಂಕ std::vector<vk_tile_config>& tc_ತಳ,
                                 ಸ್ಥಿರಾಂಕ std::string& name, size_t len, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ* ದತ್ತಾಂಶ, uint32_t pc_ಗಾತ್ರ, uint32_t pc,
                                 uint32_t rsgs = 0) {
            ಸ್ವಯಂಚಾಲಿತ tc = filter_tc(tc_ತಳ, ಕೀಲಿ.type_a, ಕೀಲಿ.mul_mat_id, ನಿಜ);
            ಆದರೆ (!tc.ಖಾಲಿಗಳು()) {
                spec_fn_t ತಟಸ್ಥ = [](ಸ್ಥಿರಾಂಕ std::vector<uint32_t>& wt, ತಾರ್ಕಿಕ) { ಹಿಂತಿರುಗಿ wt; };
                create_mm_pipelines(ಕೀಲಿ, tc, name, len, ದತ್ತಾಂಶ, pc_ಗಾತ್ರ, pc, ತಟಸ್ಥ, ಸುಳ್ಳು, rsgs > 0, rsgs, ಸುಳ್ಳು);
            }
        };
#ಆದರೆ_ಅಂತ್ಯ

        std::vector<vk_tile_config> tc_id = {{s_warptile_id, s_wg_denoms, s_align}, {m_warptile_id, m_wg_denoms, m_align}, {l_warptile_id, l_wg_denoms, l_align}};
        std::vector<vk_tile_config> tc_mmqid = {{s_warptile_mmqid, s_mmq_wg_denoms, s_align}, {m_warptile_mmqid, m_mmq_wg_denoms, m_align}, {l_warptile_mmqid, l_mmq_wg_denoms, l_align}};

        ಆದರೆ (device->fp16) {
            // FP16 subgroup path - with dot2 runtime selection
            #ವ್ಯಾಖ್ಯಾನಿಸು SPV_DOT2(NAME) (device->dot2_f16 ? NAME ## _dot2_len : NAME ## _len), (device->dot2_f16 ? NAME ## dot2_ದತ್ತಾಂಶ : NAME ## ದತ್ತಾಂಶ_2)
            #ವ್ಯಾಖ್ಯಾನಿಸು SPV_DOT2_F16ACC(NAME) (device->dot2_f16 ? NAME ## _dot2_f16acc_len : NAME ## _f16acc_len), (device->dot2_f16 ? NAME ## dot2_f16acc_ದತ್ತಾಂಶ : NAME ## f16acc_ದತ್ತಾಂಶ)

            sg_create({GGML_TYPE_F32, GGML_TYPE_F32, ಸುಳ್ಳು, ಸುಳ್ಳು}, tc_mm, "matmul_f32_f32", SPV_DOT2(matmul_f32_f32), ಗಾತ್ರ(vk_mat_mat_ತಳ್ಳು_constants), 3);
            sg_create({GGML_TYPE_F32, GGML_TYPE_F16, ಸುಳ್ಳು, ಸುಳ್ಳು}, tc_mm, "matmul_f32_f16", SPV_DOT2(matmul_f32_f16), ಗಾತ್ರ(vk_mat_mat_ತಳ್ಳು_constants), 3);
            sg_create({GGML_TYPE_F16, GGML_TYPE_F16, ಸುಳ್ಳು, ನಿಜ},  tc_mm, "matmul_f16_f16acc",     SPV_DOT2_F16ACC(matmul_f16),     ಗಾತ್ರ(vk_mat_mat_ತಳ್ಳು_constants), 3);
            sg_create({GGML_TYPE_F16, GGML_TYPE_F16, ಸುಳ್ಳು, ಸುಳ್ಳು}, tc_mm, "matmul_f16",            SPV_DOT2(matmul_f16),            ಗಾತ್ರ(vk_mat_mat_ತಳ್ಳು_constants), 3);
            sg_create({GGML_TYPE_F16, GGML_TYPE_F32, ಸುಳ್ಳು, ನಿಜ},  tc_mm, "matmul_f16_f32_f16acc", SPV_DOT2_F16ACC(matmul_f16_f32), ಗಾತ್ರ(vk_mat_mat_ತಳ್ಳು_constants), 3);
            sg_create({GGML_TYPE_F16, GGML_TYPE_F32, ಸುಳ್ಳು, ಸುಳ್ಳು}, tc_mm, "matmul_f16_f32",        SPV_DOT2(matmul_f16_f32),        ಗಾತ್ರ(vk_mat_mat_ತಳ್ಳು_constants), 3);
            // BF16 - no dot2
            sg_create({GGML_TYPE_BF16, GGML_TYPE_BF16, ಸುಳ್ಳು, ಸುಳ್ಳು}, tc_mm, "matmul_bf16", matmul_bf16_len, matmul_bf16_ದತ್ತಾಂಶ, ಗಾತ್ರ(vk_mat_mat_ತಳ್ಳು_constants), 3);

            ಗಾಗಿ (ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ type : non_lut_quant_types) {
                sg_create_quant({type, GGML_TYPE_F32, ಸುಳ್ಳು, ನಿಜ},  tc_mmq, "matmul_quant_f32_f16acc", SPV_DOT2_F16ACC(matmul_quant_f32), ಗಾತ್ರ(vk_mat_mat_ತಳ್ಳು_constants), 3);
                sg_create_quant({type, GGML_TYPE_F32, ಸುಳ್ಳು, ಸುಳ್ಳು}, tc_mmq, "matmul_quant_f32",        SPV_DOT2(matmul_quant_f32),        ಗಾತ್ರ(vk_mat_mat_ತಳ್ಳು_constants), 3);
            }
    #ವ್ಯಾಖ್ಯಾನಿಸು X_SG(TYPE, tstr) \
            sg_create({TYPE, GGML_TYPE_F32, ಸುಳ್ಳು, ನಿಜ},  tc_mmq, "matmul_" #tstr "_f32_f16acc", SPV_DOT2_F16ACC(matmul_##tstr##_f32), ಗಾತ್ರ(vk_mat_mat_ತಳ್ಳು_constants), 3); \
            sg_create({TYPE, GGML_TYPE_F32, ಸುಳ್ಳು, ಸುಳ್ಳು}, tc_mmq, "matmul_" #tstr "_f32",        SPV_DOT2(matmul_##tstr##_f32),        ಗಾತ್ರ(vk_mat_mat_ತಳ್ಳು_constants), 3);
            FOR_EACH_LUT_TYPE(X_SG)
#ವ್ಯಾಖ್ಯಾನ_ರದ್ದು X_SG

#ಆದರೆ defined(GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT)
            ಆದರೆ (device->integer_dot_ಗುಣಲಬ್ಧ) {
                std::vector<vk_tile_config> tc_mmq_int = {{s_warptile_mmq_int, s_mmq_wg_denoms, s_align}, {m_warptile_mmq_int, m_mmq_wg_denoms, m_align}, {l_warptile_mmq_int, l_mmq_wg_denoms, l_align}};
                std::vector<vk_tile_config> tc_mmq_int_k = {{s_warptile_mmq_int_k, s_mmq_wg_denoms, s_align}, {m_warptile_mmq_int_k, m_mmq_wg_denoms, m_align}, {l_warptile_mmq_int_k, l_mmq_wg_denoms, l_align}};
                sg_create_mmq({GGML_TYPE_Q2_0, GGML_TYPE_Q8_1, ಸುಳ್ಳು, ಸುಳ್ಳು}, tc_mmq_int, "matmul_q2_0_q8_1", matmul_q2_0_q8_1_len, matmul_q2_0_q8_1_ದತ್ತಾಂಶ, ಗಾತ್ರ(vk_mat_mat_ತಳ್ಳು_constants), 3);
                sg_create_mmq({GGML_TYPE_Q4_0, GGML_TYPE_Q8_1, ಸುಳ್ಳು, ಸುಳ್ಳು}, tc_mmq_int, "matmul_q4_0_q8_1", matmul_q4_0_q8_1_len, matmul_q4_0_q8_1_ದತ್ತಾಂಶ, ಗಾತ್ರ(vk_mat_mat_ತಳ್ಳು_constants), 3);
                sg_create_mmq({GGML_TYPE_Q4_1, GGML_TYPE_Q8_1, ಸುಳ್ಳು, ಸುಳ್ಳು}, tc_mmq_int, "matmul_q4_1_q8_1", matmul_q4_1_q8_1_len, matmul_q4_1_q8_1_ದತ್ತಾಂಶ, ಗಾತ್ರ(vk_mat_mat_ತಳ್ಳು_constants), 3);
                sg_create_mmq({GGML_TYPE_Q5_0, GGML_TYPE_Q8_1, ಸುಳ್ಳು, ಸುಳ್ಳು}, tc_mmq_int, "matmul_q5_0_q8_1", matmul_q5_0_q8_1_len, matmul_q5_0_q8_1_ದತ್ತಾಂಶ, ಗಾತ್ರ(vk_mat_mat_ತಳ್ಳು_constants), 3);
                sg_create_mmq({GGML_TYPE_Q5_1, GGML_TYPE_Q8_1, ಸುಳ್ಳು, ಸುಳ್ಳು}, tc_mmq_int, "matmul_q5_1_q8_1", matmul_q5_1_q8_1_len, matmul_q5_1_q8_1_ದತ್ತಾಂಶ, ಗಾತ್ರ(vk_mat_mat_ತಳ್ಳು_constants), 3);
                sg_create_mmq({GGML_TYPE_Q8_0, GGML_TYPE_Q8_1, ಸುಳ್ಳು, ಸುಳ್ಳು}, tc_mmq_int, "matmul_q8_0_q8_1", matmul_q8_0_q8_1_len, matmul_q8_0_q8_1_ದತ್ತಾಂಶ, ಗಾತ್ರ(vk_mat_mat_ತಳ್ಳು_constants), 3);
                sg_create_mmq({GGML_TYPE_MXFP4, GGML_TYPE_Q8_1, ಸುಳ್ಳು, ಸುಳ್ಳು}, tc_mmq_int, "matmul_mxfp4_q8_1", matmul_mxfp4_q8_1_len, matmul_mxfp4_q8_1_ದತ್ತಾಂಶ, ಗಾತ್ರ(vk_mat_mat_ತಳ್ಳು_constants), 3);
                sg_create_mmq({GGML_TYPE_IQ4_XS, GGML_TYPE_Q8_1, ಸುಳ್ಳು, ಸುಳ್ಳು}, tc_mmq_int, "matmul_iq4_xs_q8_1", matmul_iq4_xs_q8_1_len, matmul_iq4_xs_q8_1_ದತ್ತಾಂಶ, ಗಾತ್ರ(vk_mat_mat_ತಳ್ಳು_constants), 3);
                sg_create_mmq({GGML_TYPE_Q2_K, GGML_TYPE_Q8_1, ಸುಳ್ಳು, ಸುಳ್ಳು}, tc_mmq_int_k, "matmul_q2_k_q8_1", matmul_q2_k_q8_1_len, matmul_q2_k_q8_1_ದತ್ತಾಂಶ, ಗಾತ್ರ(vk_mat_mat_ತಳ್ಳು_constants), 3);
                sg_create_mmq({GGML_TYPE_Q3_K, GGML_TYPE_Q8_1, ಸುಳ್ಳು, ಸುಳ್ಳು}, tc_mmq_int_k, "matmul_q3_k_q8_1", matmul_q3_k_q8_1_len, matmul_q3_k_q8_1_ದತ್ತಾಂಶ, ಗಾತ್ರ(vk_mat_mat_ತಳ್ಳು_constants), 3);
                sg_create_mmq({GGML_TYPE_Q4_K, GGML_TYPE_Q8_1, ಸುಳ್ಳು, ಸುಳ್ಳು}, tc_mmq_int_k, "matmul_q4_k_q8_1", matmul_q4_k_q8_1_len, matmul_q4_k_q8_1_ದತ್ತಾಂಶ, ಗಾತ್ರ(vk_mat_mat_ತಳ್ಳು_constants), 3);
                sg_create_mmq({GGML_TYPE_Q5_K, GGML_TYPE_Q8_1, ಸುಳ್ಳು, ಸುಳ್ಳು}, tc_mmq_int_k, "matmul_q5_k_q8_1", matmul_q5_k_q8_1_len, matmul_q5_k_q8_1_ದತ್ತಾಂಶ, ಗಾತ್ರ(vk_mat_mat_ತಳ್ಳು_constants), 3);
                sg_create_mmq({GGML_TYPE_Q6_K, GGML_TYPE_Q8_1, ಸುಳ್ಳು, ಸುಳ್ಳು}, tc_mmq_int_k, "matmul_q6_k_q8_1", matmul_q6_k_q8_1_len, matmul_q6_k_q8_1_ದತ್ತಾಂಶ, ಗಾತ್ರ(vk_mat_mat_ತಳ್ಳು_constants), 3);
                sg_create_mmq({GGML_TYPE_IQ3_S, GGML_TYPE_Q8_1, ಸುಳ್ಳು, ಸುಳ್ಳು}, tc_mmq_int_k, "matmul_iq3_s_q8_1", matmul_iq3_s_q8_1_len, matmul_iq3_s_q8_1_ದತ್ತಾಂಶ, ಗಾತ್ರ(vk_mat_mat_ತಳ್ಳು_constants), 3);
            }
#ಆದರೆ_ಅಂತ್ಯ

            ಆದರೆ (device->subgroup_ballot && device->subgroup_require_ತುಂಬಿದ_support && subgroup_min_ಗಾತ್ರ_16) {
                sg_create({GGML_TYPE_F32, GGML_TYPE_F32, ನಿಜ, ಸುಳ್ಳು}, tc_id, "matmul_id_subgroup_f32_f32", SPV_DOT2(matmul_id_subgroup_f32_f32), ಗಾತ್ರ(vk_mat_mat_id_ತಳ್ಳು_constants), mul_mat_id_param_ಎಣಿಕೆ, mul_mat_subgroup_ಗಾತ್ರ_16);
                sg_create({GGML_TYPE_F16, GGML_TYPE_F16, ನಿಜ, ನಿಜ},  tc_id, "matmul_id_subgroup_f16_f16acc",     SPV_DOT2_F16ACC(matmul_id_subgroup_f16),     ಗಾತ್ರ(vk_mat_mat_id_ತಳ್ಳು_constants), mul_mat_id_param_ಎಣಿಕೆ, mul_mat_subgroup_ಗಾತ್ರ_16);
                sg_create({GGML_TYPE_F16, GGML_TYPE_F16, ನಿಜ, ಸುಳ್ಳು}, tc_id, "matmul_id_subgroup_f16",            SPV_DOT2(matmul_id_subgroup_f16),            ಗಾತ್ರ(vk_mat_mat_id_ತಳ್ಳು_constants), mul_mat_id_param_ಎಣಿಕೆ, mul_mat_subgroup_ಗಾತ್ರ_16);
                sg_create({GGML_TYPE_F16, GGML_TYPE_F32, ನಿಜ, ನಿಜ},  tc_id, "matmul_id_subgroup_f16_f32_f16acc", SPV_DOT2_F16ACC(matmul_id_subgroup_f16_f32), ಗಾತ್ರ(vk_mat_mat_id_ತಳ್ಳು_constants), mul_mat_id_param_ಎಣಿಕೆ, mul_mat_subgroup_ಗಾತ್ರ_16);
                sg_create({GGML_TYPE_F16, GGML_TYPE_F32, ನಿಜ, ಸುಳ್ಳು}, tc_id, "matmul_id_subgroup_f16_f32",        SPV_DOT2(matmul_id_subgroup_f16_f32),        ಗಾತ್ರ(vk_mat_mat_id_ತಳ್ಳು_constants), mul_mat_id_param_ಎಣಿಕೆ, mul_mat_subgroup_ಗಾತ್ರ_16);
                // BF16 id - no dot2
                sg_create({GGML_TYPE_BF16, GGML_TYPE_BF16, ನಿಜ, ಸುಳ್ಳು}, tc_id, "matmul_id_subgroup_bf16", matmul_id_subgroup_bf16_len, matmul_id_subgroup_bf16_ದತ್ತಾಂಶ, ಗಾತ್ರ(vk_mat_mat_id_ತಳ್ಳು_constants), mul_mat_id_param_ಎಣಿಕೆ, mul_mat_subgroup_ಗಾತ್ರ_16);
                ಗಾಗಿ (ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ type : non_lut_quant_types) {
                    sg_create_quant({type, GGML_TYPE_F32, ನಿಜ, ನಿಜ},  tc_mmqid, "matmul_id_subgroup_quant_f32_f16acc", SPV_DOT2_F16ACC(matmul_id_subgroup_quant_f32), ಗಾತ್ರ(vk_mat_mat_id_ತಳ್ಳು_constants), mul_mat_id_param_ಎಣಿಕೆ, mul_mat_subgroup_ಗಾತ್ರ);
                    sg_create_quant({type, GGML_TYPE_F32, ನಿಜ, ಸುಳ್ಳು}, tc_mmqid, "matmul_id_subgroup_quant_f32",        SPV_DOT2(matmul_id_subgroup_quant_f32),        ಗಾತ್ರ(vk_mat_mat_id_ತಳ್ಳು_constants), mul_mat_id_param_ಎಣಿಕೆ, mul_mat_subgroup_ಗಾತ್ರ);
                }
        #ವ್ಯಾಖ್ಯಾನಿಸು X_SG_ID_SUB(TYPE, tstr) \
                sg_create({TYPE, GGML_TYPE_F32, ನಿಜ, ನಿಜ},  tc_mmqid, "matmul_id_subgroup_" #tstr "_f32_f16acc", SPV_DOT2_F16ACC(matmul_id_subgroup_##tstr##_f32), ಗಾತ್ರ(vk_mat_mat_id_ತಳ್ಳು_constants), mul_mat_id_param_ಎಣಿಕೆ, mul_mat_subgroup_ಗಾತ್ರ); \
                sg_create({TYPE, GGML_TYPE_F32, ನಿಜ, ಸುಳ್ಳು}, tc_mmqid, "matmul_id_subgroup_" #tstr "_f32",        SPV_DOT2(matmul_id_subgroup_##tstr##_f32),        ಗಾತ್ರ(vk_mat_mat_id_ತಳ್ಳು_constants), mul_mat_id_param_ಎಣಿಕೆ, mul_mat_subgroup_ಗಾತ್ರ);
                FOR_EACH_LUT_TYPE(X_SG_ID_SUB)
#ವ್ಯಾಖ್ಯಾನ_ರದ್ದು X_SG_ID_SUB
#ಆದರೆ defined(GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT)
                ಆದರೆ (device->integer_dot_ಗುಣಲಬ್ಧ) {
                    std::vector<vk_tile_config> tc_mmqid_int = {{s_warptile_mmqid_int, s_mmq_wg_denoms, s_align}, {m_warptile_mmqid_int, m_mmq_wg_denoms, m_align}, {l_warptile_mmqid_int, l_mmq_wg_denoms, l_align}};
                    std::vector<vk_tile_config> tc_mmqid_int_k = {{s_warptile_mmqid_int_k, s_mmq_wg_denoms, s_align}, {m_warptile_mmqid_int_k, m_mmq_wg_denoms, m_align}, {l_warptile_mmqid_int_k, l_mmq_wg_denoms, l_align}};
                    sg_create_mmq({GGML_TYPE_Q2_0, GGML_TYPE_Q8_1, ನಿಜ, ಸುಳ್ಳು}, tc_mmqid_int, "matmul_id_subgroup_q2_0_q8_1", matmul_id_subgroup_q2_0_q8_1_len, matmul_id_subgroup_q2_0_q8_1_ದತ್ತಾಂಶ, ಗಾತ್ರ(vk_mat_mat_id_ತಳ್ಳು_constants), mul_mat_id_param_ಎಣಿಕೆ, mul_mat_subgroup_ಗಾತ್ರ);
                    sg_create_mmq({GGML_TYPE_Q4_0, GGML_TYPE_Q8_1, ನಿಜ, ಸುಳ್ಳು}, tc_mmqid_int, "matmul_id_subgroup_q4_0_q8_1", matmul_id_subgroup_q4_0_q8_1_len, matmul_id_subgroup_q4_0_q8_1_ದತ್ತಾಂಶ, ಗಾತ್ರ(vk_mat_mat_id_ತಳ್ಳು_constants), mul_mat_id_param_ಎಣಿಕೆ, mul_mat_subgroup_ಗಾತ್ರ);
                    sg_create_mmq({GGML_TYPE_Q4_1, GGML_TYPE_Q8_1, ನಿಜ, ಸುಳ್ಳು}, tc_mmqid_int, "matmul_id_subgroup_q4_1_q8_1", matmul_id_subgroup_q4_1_q8_1_len, matmul_id_subgroup_q4_1_q8_1_ದತ್ತಾಂಶ, ಗಾತ್ರ(vk_mat_mat_id_ತಳ್ಳು_constants), mul_mat_id_param_ಎಣಿಕೆ, mul_mat_subgroup_ಗಾತ್ರ);
                    sg_create_mmq({GGML_TYPE_Q5_0, GGML_TYPE_Q8_1, ನಿಜ, ಸುಳ್ಳು}, tc_mmqid_int, "matmul_id_subgroup_q5_0_q8_1", matmul_id_subgroup_q5_0_q8_1_len, matmul_id_subgroup_q5_0_q8_1_ದತ್ತಾಂಶ, ಗಾತ್ರ(vk_mat_mat_id_ತಳ್ಳು_constants), mul_mat_id_param_ಎಣಿಕೆ, mul_mat_subgroup_ಗಾತ್ರ);
                    sg_create_mmq({GGML_TYPE_Q5_1, GGML_TYPE_Q8_1, ನಿಜ, ಸುಳ್ಳು}, tc_mmqid_int, "matmul_id_subgroup_q5_1_q8_1", matmul_id_subgroup_q5_1_q8_1_len, matmul_id_subgroup_q5_1_q8_1_ದತ್ತಾಂಶ, ಗಾತ್ರ(vk_mat_mat_id_ತಳ್ಳು_constants), mul_mat_id_param_ಎಣಿಕೆ, mul_mat_subgroup_ಗಾತ್ರ);
                    sg_create_mmq({GGML_TYPE_Q8_0, GGML_TYPE_Q8_1, ನಿಜ, ಸುಳ್ಳು}, tc_mmqid_int, "matmul_id_subgroup_q8_0_q8_1", matmul_id_subgroup_q8_0_q8_1_len, matmul_id_subgroup_q8_0_q8_1_ದತ್ತಾಂಶ, ಗಾತ್ರ(vk_mat_mat_id_ತಳ್ಳು_constants), mul_mat_id_param_ಎಣಿಕೆ, mul_mat_subgroup_ಗಾತ್ರ);
                    sg_create_mmq({GGML_TYPE_MXFP4, GGML_TYPE_Q8_1, ನಿಜ, ಸುಳ್ಳು}, tc_mmqid_int, "matmul_id_subgroup_mxfp4_q8_1", matmul_id_subgroup_mxfp4_q8_1_len, matmul_id_subgroup_mxfp4_q8_1_ದತ್ತಾಂಶ, ಗಾತ್ರ(vk_mat_mat_id_ತಳ್ಳು_constants), mul_mat_id_param_ಎಣಿಕೆ, mul_mat_subgroup_ಗಾತ್ರ);
                    sg_create_mmq({GGML_TYPE_IQ4_XS, GGML_TYPE_Q8_1, ನಿಜ, ಸುಳ್ಳು}, tc_mmqid_int, "matmul_id_subgroup_iq4_xs_q8_1", matmul_id_subgroup_iq4_xs_q8_1_len, matmul_id_subgroup_iq4_xs_q8_1_ದತ್ತಾಂಶ, ಗಾತ್ರ(vk_mat_mat_id_ತಳ್ಳು_constants), mul_mat_id_param_ಎಣಿಕೆ, mul_mat_subgroup_ಗಾತ್ರ);
                    sg_create_mmq({GGML_TYPE_Q2_K, GGML_TYPE_Q8_1, ನಿಜ, ಸುಳ್ಳು}, tc_mmqid_int_k, "matmul_id_subgroup_q2_k_q8_1", matmul_id_subgroup_q2_k_q8_1_len, matmul_id_subgroup_q2_k_q8_1_ದತ್ತಾಂಶ, ಗಾತ್ರ(vk_mat_mat_id_ತಳ್ಳು_constants), mul_mat_id_param_ಎಣಿಕೆ, mul_mat_subgroup_ಗಾತ್ರ_16);
                    sg_create_mmq({GGML_TYPE_Q3_K, GGML_TYPE_Q8_1, ನಿಜ, ಸುಳ್ಳು}, tc_mmqid_int_k, "matmul_id_subgroup_q3_k_q8_1", matmul_id_subgroup_q3_k_q8_1_len, matmul_id_subgroup_q3_k_q8_1_ದತ್ತಾಂಶ, ಗಾತ್ರ(vk_mat_mat_id_ತಳ್ಳು_constants), mul_mat_id_param_ಎಣಿಕೆ, mul_mat_subgroup_ಗಾತ್ರ_16);
                    sg_create_mmq({GGML_TYPE_Q4_K, GGML_TYPE_Q8_1, ನಿಜ, ಸುಳ್ಳು}, tc_mmqid_int_k, "matmul_id_subgroup_q4_k_q8_1", matmul_id_subgroup_q4_k_q8_1_len, matmul_id_subgroup_q4_k_q8_1_ದತ್ತಾಂಶ, ಗಾತ್ರ(vk_mat_mat_id_ತಳ್ಳು_constants), mul_mat_id_param_ಎಣಿಕೆ, mul_mat_subgroup_ಗಾತ್ರ_16);
                    sg_create_mmq({GGML_TYPE_Q5_K, GGML_TYPE_Q8_1, ನಿಜ, ಸುಳ್ಳು}, tc_mmqid_int_k, "matmul_id_subgroup_q5_k_q8_1", matmul_id_subgroup_q5_k_q8_1_len, matmul_id_subgroup_q5_k_q8_1_ದತ್ತಾಂಶ, ಗಾತ್ರ(vk_mat_mat_id_ತಳ್ಳು_constants), mul_mat_id_param_ಎಣಿಕೆ, mul_mat_subgroup_ಗಾತ್ರ_16);
                    sg_create_mmq({GGML_TYPE_Q6_K, GGML_TYPE_Q8_1, ನಿಜ, ಸುಳ್ಳು}, tc_mmqid_int_k, "matmul_id_subgroup_q6_k_q8_1", matmul_id_subgroup_q6_k_q8_1_len, matmul_id_subgroup_q6_k_q8_1_ದತ್ತಾಂಶ, ಗಾತ್ರ(vk_mat_mat_id_ತಳ್ಳು_constants), mul_mat_id_param_ಎಣಿಕೆ, mul_mat_subgroup_ಗಾತ್ರ_16);
                    sg_create_mmq({GGML_TYPE_IQ3_S, GGML_TYPE_Q8_1, ನಿಜ, ಸುಳ್ಳು}, tc_mmqid_int_k, "matmul_id_subgroup_iq3_s_q8_1", matmul_id_subgroup_iq3_s_q8_1_len, matmul_id_subgroup_iq3_s_q8_1_ದತ್ತಾಂಶ, ಗಾತ್ರ(vk_mat_mat_id_ತಳ್ಳು_constants), mul_mat_id_param_ಎಣಿಕೆ, mul_mat_subgroup_ಗಾತ್ರ_16);
                }
#ಆದರೆ_ಅಂತ್ಯ
            } ಇಲ್ಲದಿದ್ದರೆ {
                sg_create({GGML_TYPE_F32, GGML_TYPE_F32, ನಿಜ, ಸುಳ್ಳು}, tc_mm, "matmul_id_f32_f32", SPV_DOT2(matmul_id_f32_f32), ಗಾತ್ರ(vk_mat_mat_id_ತಳ್ಳು_constants), mul_mat_id_param_ಎಣಿಕೆ);
                sg_create({GGML_TYPE_F16, GGML_TYPE_F16, ನಿಜ, ನಿಜ},  tc_mm, "matmul_id_f16_f16acc",     SPV_DOT2_F16ACC(matmul_id_f16),     ಗಾತ್ರ(vk_mat_mat_id_ತಳ್ಳು_constants), mul_mat_id_param_ಎಣಿಕೆ);
                sg_create({GGML_TYPE_F16, GGML_TYPE_F16, ನಿಜ, ಸುಳ್ಳು}, tc_mm, "matmul_id_f16",            SPV_DOT2(matmul_id_f16),            ಗಾತ್ರ(vk_mat_mat_id_ತಳ್ಳು_constants), mul_mat_id_param_ಎಣಿಕೆ);
                sg_create({GGML_TYPE_F16, GGML_TYPE_F32, ನಿಜ, ನಿಜ},  tc_mm, "matmul_id_f16_f32_f16acc", SPV_DOT2_F16ACC(matmul_id_f16_f32), ಗಾತ್ರ(vk_mat_mat_id_ತಳ್ಳು_constants), mul_mat_id_param_ಎಣಿಕೆ);
                sg_create({GGML_TYPE_F16, GGML_TYPE_F32, ನಿಜ, ಸುಳ್ಳು}, tc_mm, "matmul_id_f16_f32",        SPV_DOT2(matmul_id_f16_f32),        ಗಾತ್ರ(vk_mat_mat_id_ತಳ್ಳು_constants), mul_mat_id_param_ಎಣಿಕೆ);
                // BF16 id - no dot2
                sg_create({GGML_TYPE_BF16, GGML_TYPE_BF16, ನಿಜ, ಸುಳ್ಳು}, tc_mm, "matmul_id_bf16", matmul_id_bf16_len, matmul_id_bf16_ದತ್ತಾಂಶ, ಗಾತ್ರ(vk_mat_mat_id_ತಳ್ಳು_constants), mul_mat_id_param_ಎಣಿಕೆ);
                ಗಾಗಿ (ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ type : non_lut_quant_types) {
                    sg_create_quant({type, GGML_TYPE_F32, ನಿಜ, ನಿಜ},  tc_mmqid, "matmul_id_quant_f32_f16acc", SPV_DOT2_F16ACC(matmul_id_quant_f32), ಗಾತ್ರ(vk_mat_mat_id_ತಳ್ಳು_constants), mul_mat_id_param_ಎಣಿಕೆ);
                    sg_create_quant({type, GGML_TYPE_F32, ನಿಜ, ಸುಳ್ಳು}, tc_mmqid, "matmul_id_quant_f32",        SPV_DOT2(matmul_id_quant_f32),        ಗಾತ್ರ(vk_mat_mat_id_ತಳ್ಳು_constants), mul_mat_id_param_ಎಣಿಕೆ);
                }
        #ವ್ಯಾಖ್ಯಾನಿಸು X_SG_ID(TYPE, tstr) \
                sg_create({TYPE, GGML_TYPE_F32, ನಿಜ, ನಿಜ},  tc_mmqid, "matmul_id_" #tstr "_f32_f16acc", SPV_DOT2_F16ACC(matmul_id_##tstr##_f32), ಗಾತ್ರ(vk_mat_mat_id_ತಳ್ಳು_constants), mul_mat_id_param_ಎಣಿಕೆ); \
                sg_create({TYPE, GGML_TYPE_F32, ನಿಜ, ಸುಳ್ಳು}, tc_mmqid, "matmul_id_" #tstr "_f32",        SPV_DOT2(matmul_id_##tstr##_f32),        ಗಾತ್ರ(vk_mat_mat_id_ತಳ್ಳು_constants), mul_mat_id_param_ಎಣಿಕೆ);
                FOR_EACH_LUT_TYPE(X_SG_ID)
#ವ್ಯಾಖ್ಯಾನ_ರದ್ದು X_SG_ID
#ಆದರೆ defined(GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT)
                ಆದರೆ (device->integer_dot_ಗುಣಲಬ್ಧ) {
                    std::vector<vk_tile_config> tc_mmqid_int = {{s_warptile_mmqid_int, s_mmq_wg_denoms, s_align}, {m_warptile_mmqid_int, m_mmq_wg_denoms, m_align}, {l_warptile_mmqid_int, l_mmq_wg_denoms, l_align}};
                    std::vector<vk_tile_config> tc_mmqid_int_k = {{s_warptile_mmqid_int_k, s_mmq_wg_denoms, s_align}, {m_warptile_mmqid_int_k, m_mmq_wg_denoms, m_align}, {l_warptile_mmqid_int_k, l_mmq_wg_denoms, l_align}};
                    sg_create_mmq({GGML_TYPE_Q2_0, GGML_TYPE_Q8_1, ನಿಜ, ಸುಳ್ಳು}, tc_mmqid_int, "matmul_id_q2_0_q8_1", matmul_id_q2_0_q8_1_len, matmul_id_q2_0_q8_1_ದತ್ತಾಂಶ, ಗಾತ್ರ(vk_mat_mat_id_ತಳ್ಳು_constants), mul_mat_id_param_ಎಣಿಕೆ);
                    sg_create_mmq({GGML_TYPE_Q4_0, GGML_TYPE_Q8_1, ನಿಜ, ಸುಳ್ಳು}, tc_mmqid_int, "matmul_id_q4_0_q8_1", matmul_id_q4_0_q8_1_len, matmul_id_q4_0_q8_1_ದತ್ತಾಂಶ, ಗಾತ್ರ(vk_mat_mat_id_ತಳ್ಳು_constants), mul_mat_id_param_ಎಣಿಕೆ);
                    sg_create_mmq({GGML_TYPE_Q4_1, GGML_TYPE_Q8_1, ನಿಜ, ಸುಳ್ಳು}, tc_mmqid_int, "matmul_id_q4_1_q8_1", matmul_id_q4_1_q8_1_len, matmul_id_q4_1_q8_1_ದತ್ತಾಂಶ, ಗಾತ್ರ(vk_mat_mat_id_ತಳ್ಳು_constants), mul_mat_id_param_ಎಣಿಕೆ);
                    sg_create_mmq({GGML_TYPE_Q5_0, GGML_TYPE_Q8_1, ನಿಜ, ಸುಳ್ಳು}, tc_mmqid_int, "matmul_id_q5_0_q8_1", matmul_id_q5_0_q8_1_len, matmul_id_q5_0_q8_1_ದತ್ತಾಂಶ, ಗಾತ್ರ(vk_mat_mat_id_ತಳ್ಳು_constants), mul_mat_id_param_ಎಣಿಕೆ);
                    sg_create_mmq({GGML_TYPE_Q5_1, GGML_TYPE_Q8_1, ನಿಜ, ಸುಳ್ಳು}, tc_mmqid_int, "matmul_id_q5_1_q8_1", matmul_id_q5_1_q8_1_len, matmul_id_q5_1_q8_1_ದತ್ತಾಂಶ, ಗಾತ್ರ(vk_mat_mat_id_ತಳ್ಳು_constants), mul_mat_id_param_ಎಣಿಕೆ);
                    sg_create_mmq({GGML_TYPE_Q8_0, GGML_TYPE_Q8_1, ನಿಜ, ಸುಳ್ಳು}, tc_mmqid_int, "matmul_id_q8_0_q8_1", matmul_id_q8_0_q8_1_len, matmul_id_q8_0_q8_1_ದತ್ತಾಂಶ, ಗಾತ್ರ(vk_mat_mat_id_ತಳ್ಳು_constants), mul_mat_id_param_ಎಣಿಕೆ);
                    sg_create_mmq({GGML_TYPE_MXFP4, GGML_TYPE_Q8_1, ನಿಜ, ಸುಳ್ಳು}, tc_mmqid_int, "matmul_id_mxfp4_q8_1", matmul_id_mxfp4_q8_1_len, matmul_id_mxfp4_q8_1_ದತ್ತಾಂಶ, ಗಾತ್ರ(vk_mat_mat_id_ತಳ್ಳು_constants), mul_mat_id_param_ಎಣಿಕೆ);
                    sg_create_mmq({GGML_TYPE_IQ4_XS, GGML_TYPE_Q8_1, ನಿಜ, ಸುಳ್ಳು}, tc_mmqid_int, "matmul_id_iq4_xs_q8_1", matmul_id_iq4_xs_q8_1_len, matmul_id_iq4_xs_q8_1_ದತ್ತಾಂಶ, ಗಾತ್ರ(vk_mat_mat_id_ತಳ್ಳು_constants), mul_mat_id_param_ಎಣಿಕೆ);
                    sg_create_mmq({GGML_TYPE_Q2_K, GGML_TYPE_Q8_1, ನಿಜ, ಸುಳ್ಳು}, tc_mmqid_int_k, "matmul_id_q2_k_q8_1", matmul_id_q2_k_q8_1_len, matmul_id_q2_k_q8_1_ದತ್ತಾಂಶ, ಗಾತ್ರ(vk_mat_mat_id_ತಳ್ಳು_constants), mul_mat_id_param_ಎಣಿಕೆ);
                    sg_create_mmq({GGML_TYPE_Q3_K, GGML_TYPE_Q8_1, ನಿಜ, ಸುಳ್ಳು}, tc_mmqid_int_k, "matmul_id_q3_k_q8_1", matmul_id_q3_k_q8_1_len, matmul_id_q3_k_q8_1_ದತ್ತಾಂಶ, ಗಾತ್ರ(vk_mat_mat_id_ತಳ್ಳು_constants), mul_mat_id_param_ಎಣಿಕೆ);
                    sg_create_mmq({GGML_TYPE_Q4_K, GGML_TYPE_Q8_1, ನಿಜ, ಸುಳ್ಳು}, tc_mmqid_int_k, "matmul_id_q4_k_q8_1", matmul_id_q4_k_q8_1_len, matmul_id_q4_k_q8_1_ದತ್ತಾಂಶ, ಗಾತ್ರ(vk_mat_mat_id_ತಳ್ಳು_constants), mul_mat_id_param_ಎಣಿಕೆ);
                    sg_create_mmq({GGML_TYPE_Q5_K, GGML_TYPE_Q8_1, ನಿಜ, ಸುಳ್ಳು}, tc_mmqid_int_k, "matmul_id_q5_k_q8_1", matmul_id_q5_k_q8_1_len, matmul_id_q5_k_q8_1_ದತ್ತಾಂಶ, ಗಾತ್ರ(vk_mat_mat_id_ತಳ್ಳು_constants), mul_mat_id_param_ಎಣಿಕೆ);
                    sg_create_mmq({GGML_TYPE_Q6_K, GGML_TYPE_Q8_1, ನಿಜ, ಸುಳ್ಳು}, tc_mmqid_int_k, "matmul_id_q6_k_q8_1", matmul_id_q6_k_q8_1_len, matmul_id_q6_k_q8_1_ದತ್ತಾಂಶ, ಗಾತ್ರ(vk_mat_mat_id_ತಳ್ಳು_constants), mul_mat_id_param_ಎಣಿಕೆ);
                    sg_create_mmq({GGML_TYPE_IQ3_S, GGML_TYPE_Q8_1, ನಿಜ, ಸುಳ್ಳು}, tc_mmqid_int_k, "matmul_id_iq3_s_q8_1", matmul_id_iq3_s_q8_1_len, matmul_id_iq3_s_q8_1_ದತ್ತಾಂಶ, ಗಾತ್ರ(vk_mat_mat_id_ತಳ್ಳು_constants), mul_mat_id_param_ಎಣಿಕೆ);
                }
#ಆದರೆ_ಅಂತ್ಯ
            }
            #ವ್ಯಾಖ್ಯಾನ_ರದ್ದು SPV_DOT2
            #ವ್ಯಾಖ್ಯಾನ_ರದ್ದು SPV_DOT2_F16ACC
        } ಇಲ್ಲದಿದ್ದರೆ {
            // FP32-only fallback path
            sg_create({GGML_TYPE_F32, GGML_TYPE_F32, ಸುಳ್ಳು, ಸುಳ್ಳು}, tc_mm, "matmul_f32_f32", matmul_f32_f32_fp32_len, matmul_f32_f32_fp32_ದತ್ತಾಂಶ, ಗಾತ್ರ(vk_mat_mat_ತಳ್ಳು_constants), 3);
            sg_create({GGML_TYPE_F32, GGML_TYPE_F16, ಸುಳ್ಳು, ಸುಳ್ಳು}, tc_mm, "matmul_f32_f16", matmul_f32_f16_fp32_len, matmul_f32_f16_fp32_ದತ್ತಾಂಶ, ಗಾತ್ರ(vk_mat_mat_ತಳ್ಳು_constants), 3);
            sg_create({GGML_TYPE_F16, GGML_TYPE_F16, ಸುಳ್ಳು, ಸುಳ್ಳು}, tc_mm, "matmul_f16",     matmul_f16_fp32_len,     matmul_f16_fp32_ದತ್ತಾಂಶ,     ಗಾತ್ರ(vk_mat_mat_ತಳ್ಳು_constants), 3);
            sg_create({GGML_TYPE_F16, GGML_TYPE_F32, ಸುಳ್ಳು, ಸುಳ್ಳು}, tc_mm, "matmul_f16_f32", matmul_f16_f32_fp32_len, matmul_f16_f32_fp32_ದತ್ತಾಂಶ, ಗಾತ್ರ(vk_mat_mat_ತಳ್ಳು_constants), 3);
            sg_create({GGML_TYPE_BF16, GGML_TYPE_BF16, ಸುಳ್ಳು, ಸುಳ್ಳು}, tc_mm, "matmul_bf16",  matmul_bf16_fp32_len,    matmul_bf16_fp32_ದತ್ತಾಂಶ,    ಗಾತ್ರ(vk_mat_mat_ತಳ್ಳು_constants), 3);

            ಗಾಗಿ (ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ type : non_lut_quant_types) {
                sg_create_quant({type, GGML_TYPE_F32, ಸುಳ್ಳು, ಸುಳ್ಳು}, tc_mmq, "matmul_quant_f32", matmul_quant_f32_fp32_len, matmul_quant_f32_fp32_ದತ್ತಾಂಶ, ಗಾತ್ರ(vk_mat_mat_ತಳ್ಳು_constants), 3);
            }
    #ವ್ಯಾಖ್ಯಾನಿಸು X_SG_FP32(TYPE, tstr) \
            sg_create({TYPE, GGML_TYPE_F32, ಸುಳ್ಳು, ಸುಳ್ಳು}, tc_mmq, "matmul_" #tstr "_f32", matmul_##tstr##_f32_fp32_len, matmul_##tstr##_f32_fp32_data, ಗಾತ್ರ(vk_mat_mat_ತಳ್ಳು_constants), 3);
            FOR_EACH_LUT_TYPE(X_SG_FP32)
#ವ್ಯಾಖ್ಯಾನ_ರದ್ದು X_SG_FP32

#ಆದರೆ defined(GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT)
            ಆದರೆ (device->integer_dot_ಗುಣಲಬ್ಧ) {
                std::vector<vk_tile_config> tc_mmq_int = {{s_warptile_mmq_int, s_mmq_wg_denoms, s_align}, {m_warptile_mmq_int, m_mmq_wg_denoms, m_align}, {l_warptile_mmq_int, l_mmq_wg_denoms, l_align}};
                std::vector<vk_tile_config> tc_mmq_int_k = {{s_warptile_mmq_int_k, s_mmq_wg_denoms, s_align}, {m_warptile_mmq_int_k, m_mmq_wg_denoms, m_align}, {l_warptile_mmq_int_k, l_mmq_wg_denoms, l_align}};
                sg_create_mmq({GGML_TYPE_Q2_0, GGML_TYPE_Q8_1, ಸುಳ್ಳು, ಸುಳ್ಳು}, tc_mmq_int, "matmul_q2_0_q8_1", matmul_q2_0_q8_1_fp32_len, matmul_q2_0_q8_1_fp32_ದತ್ತಾಂಶ, ಗಾತ್ರ(vk_mat_mat_ತಳ್ಳು_constants), 3);
                sg_create_mmq({GGML_TYPE_Q4_0, GGML_TYPE_Q8_1, ಸುಳ್ಳು, ಸುಳ್ಳು}, tc_mmq_int, "matmul_q4_0_q8_1", matmul_q4_0_q8_1_fp32_len, matmul_q4_0_q8_1_fp32_ದತ್ತಾಂಶ, ಗಾತ್ರ(vk_mat_mat_ತಳ್ಳು_constants), 3);
                sg_create_mmq({GGML_TYPE_Q4_1, GGML_TYPE_Q8_1, ಸುಳ್ಳು, ಸುಳ್ಳು}, tc_mmq_int, "matmul_q4_1_q8_1", matmul_q4_1_q8_1_fp32_len, matmul_q4_1_q8_1_fp32_ದತ್ತಾಂಶ, ಗಾತ್ರ(vk_mat_mat_ತಳ್ಳು_constants), 3);
                sg_create_mmq({GGML_TYPE_Q5_0, GGML_TYPE_Q8_1, ಸುಳ್ಳು, ಸುಳ್ಳು}, tc_mmq_int, "matmul_q5_0_q8_1", matmul_q5_0_q8_1_fp32_len, matmul_q5_0_q8_1_fp32_ದತ್ತಾಂಶ, ಗಾತ್ರ(vk_mat_mat_ತಳ್ಳು_constants), 3);
                sg_create_mmq({GGML_TYPE_Q5_1, GGML_TYPE_Q8_1, ಸುಳ್ಳು, ಸುಳ್ಳು}, tc_mmq_int, "matmul_q5_1_q8_1", matmul_q5_1_q8_1_fp32_len, matmul_q5_1_q8_1_fp32_ದತ್ತಾಂಶ, ಗಾತ್ರ(vk_mat_mat_ತಳ್ಳು_constants), 3);
                sg_create_mmq({GGML_TYPE_Q8_0, GGML_TYPE_Q8_1, ಸುಳ್ಳು, ಸುಳ್ಳು}, tc_mmq_int, "matmul_q8_0_q8_1", matmul_q8_0_q8_1_fp32_len, matmul_q8_0_q8_1_fp32_ದತ್ತಾಂಶ, ಗಾತ್ರ(vk_mat_mat_ತಳ್ಳು_constants), 3);
                sg_create_mmq({GGML_TYPE_IQ4_XS, GGML_TYPE_Q8_1, ಸುಳ್ಳು, ಸುಳ್ಳು}, tc_mmq_int, "matmul_iq4_xs_q8_1", matmul_iq4_xs_q8_1_fp32_len, matmul_iq4_xs_q8_1_fp32_ದತ್ತಾಂಶ, ಗಾತ್ರ(vk_mat_mat_ತಳ್ಳು_constants), 3);
                sg_create_mmq({GGML_TYPE_Q2_K, GGML_TYPE_Q8_1, ಸುಳ್ಳು, ಸುಳ್ಳು}, tc_mmq_int_k, "matmul_q2_k_q8_1", matmul_q2_k_q8_1_fp32_len, matmul_q2_k_q8_1_fp32_ದತ್ತಾಂಶ, ಗಾತ್ರ(vk_mat_mat_ತಳ್ಳು_constants), 3);
                sg_create_mmq({GGML_TYPE_Q3_K, GGML_TYPE_Q8_1, ಸುಳ್ಳು, ಸುಳ್ಳು}, tc_mmq_int_k, "matmul_q3_k_q8_1", matmul_q3_k_q8_1_fp32_len, matmul_q3_k_q8_1_fp32_ದತ್ತಾಂಶ, ಗಾತ್ರ(vk_mat_mat_ತಳ್ಳು_constants), 3);
                sg_create_mmq({GGML_TYPE_Q4_K, GGML_TYPE_Q8_1, ಸುಳ್ಳು, ಸುಳ್ಳು}, tc_mmq_int_k, "matmul_q4_k_q8_1", matmul_q4_k_q8_1_fp32_len, matmul_q4_k_q8_1_fp32_ದತ್ತಾಂಶ, ಗಾತ್ರ(vk_mat_mat_ತಳ್ಳು_constants), 3);
                sg_create_mmq({GGML_TYPE_Q5_K, GGML_TYPE_Q8_1, ಸುಳ್ಳು, ಸುಳ್ಳು}, tc_mmq_int_k, "matmul_q5_k_q8_1", matmul_q5_k_q8_1_fp32_len, matmul_q5_k_q8_1_fp32_ದತ್ತಾಂಶ, ಗಾತ್ರ(vk_mat_mat_ತಳ್ಳು_constants), 3);
                sg_create_mmq({GGML_TYPE_Q6_K, GGML_TYPE_Q8_1, ಸುಳ್ಳು, ಸುಳ್ಳು}, tc_mmq_int_k, "matmul_q6_k_q8_1", matmul_q6_k_q8_1_fp32_len, matmul_q6_k_q8_1_fp32_ದತ್ತಾಂಶ, ಗಾತ್ರ(vk_mat_mat_ತಳ್ಳು_constants), 3);
                sg_create_mmq({GGML_TYPE_IQ3_S, GGML_TYPE_Q8_1, ಸುಳ್ಳು, ಸುಳ್ಳು}, tc_mmq_int_k, "matmul_iq3_s_q8_1", matmul_iq3_s_q8_1_fp32_len, matmul_iq3_s_q8_1_fp32_ದತ್ತಾಂಶ, ಗಾತ್ರ(vk_mat_mat_ತಳ್ಳು_constants), 3);
            }
#ಆದರೆ_ಅಂತ್ಯ

            ಆದರೆ (device->subgroup_ballot && device->subgroup_require_ತುಂಬಿದ_support && subgroup_min_ಗಾತ್ರ_16) {
                sg_create({GGML_TYPE_F32, GGML_TYPE_F32, ನಿಜ, ಸುಳ್ಳು}, tc_id, "matmul_id_subgroup_f32_f32", matmul_id_subgroup_f32_f32_fp32_len, matmul_id_subgroup_f32_f32_fp32_ದತ್ತಾಂಶ, ಗಾತ್ರ(vk_mat_mat_id_ತಳ್ಳು_constants), mul_mat_id_param_ಎಣಿಕೆ, mul_mat_subgroup_ಗಾತ್ರ_16);
                sg_create({GGML_TYPE_F16, GGML_TYPE_F16, ನಿಜ, ಸುಳ್ಳು}, tc_id, "matmul_id_subgroup_f16",     matmul_id_subgroup_f16_fp32_len,     matmul_id_subgroup_f16_fp32_ದತ್ತಾಂಶ,     ಗಾತ್ರ(vk_mat_mat_id_ತಳ್ಳು_constants), mul_mat_id_param_ಎಣಿಕೆ, mul_mat_subgroup_ಗಾತ್ರ_16);
                sg_create({GGML_TYPE_F16, GGML_TYPE_F32, ನಿಜ, ಸುಳ್ಳು}, tc_id, "matmul_id_subgroup_f16_f32", matmul_id_subgroup_f16_f32_fp32_len, matmul_id_subgroup_f16_f32_fp32_ದತ್ತಾಂಶ, ಗಾತ್ರ(vk_mat_mat_id_ತಳ್ಳು_constants), mul_mat_id_param_ಎಣಿಕೆ, mul_mat_subgroup_ಗಾತ್ರ_16);
                sg_create({GGML_TYPE_BF16, GGML_TYPE_BF16, ನಿಜ, ಸುಳ್ಳು}, tc_id, "matmul_id_subgroup_bf16",  matmul_id_subgroup_bf16_fp32_len,    matmul_id_subgroup_bf16_fp32_ದತ್ತಾಂಶ,    ಗಾತ್ರ(vk_mat_mat_id_ತಳ್ಳು_constants), mul_mat_id_param_ಎಣಿಕೆ, mul_mat_subgroup_ಗಾತ್ರ_16);
                ಗಾಗಿ (ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ type : non_lut_quant_types) {
                    sg_create_quant({type, GGML_TYPE_F32, ನಿಜ, ಸುಳ್ಳು}, tc_mmqid, "matmul_id_subgroup_quant_f32", matmul_id_subgroup_quant_f32_fp32_len, matmul_id_subgroup_quant_f32_fp32_ದತ್ತಾಂಶ, ಗಾತ್ರ(vk_mat_mat_id_ತಳ್ಳು_constants), mul_mat_id_param_ಎಣಿಕೆ, mul_mat_subgroup_ಗಾತ್ರ);
                }
        #ವ್ಯಾಖ್ಯಾನಿಸು X_SG_ID_SUB_FP32(TYPE, tstr) \
                sg_create({TYPE, GGML_TYPE_F32, ನಿಜ, ಸುಳ್ಳು}, tc_mmqid, "matmul_id_subgroup_" #tstr "_f32", matmul_id_subgroup_##tstr##_f32_fp32_len, matmul_id_subgroup_##tstr##_f32_fp32_data, ಗಾತ್ರ(vk_mat_mat_id_ತಳ್ಳು_constants), mul_mat_id_param_ಎಣಿಕೆ, mul_mat_subgroup_ಗಾತ್ರ);
                FOR_EACH_LUT_TYPE(X_SG_ID_SUB_FP32)
#ವ್ಯಾಖ್ಯಾನ_ರದ್ದು X_SG_ID_SUB_FP32
            } ಇಲ್ಲದಿದ್ದರೆ {
                sg_create({GGML_TYPE_F32, GGML_TYPE_F32, ನಿಜ, ಸುಳ್ಳು}, tc_mm, "matmul_id_f32_f32", matmul_id_f32_f32_fp32_len, matmul_id_f32_f32_fp32_ದತ್ತಾಂಶ, ಗಾತ್ರ(vk_mat_mat_id_ತಳ್ಳು_constants), mul_mat_id_param_ಎಣಿಕೆ);
                sg_create({GGML_TYPE_F16, GGML_TYPE_F16, ನಿಜ, ಸುಳ್ಳು}, tc_mm, "matmul_id_f16",     matmul_id_f16_fp32_len,     matmul_id_f16_fp32_ದತ್ತಾಂಶ,     ಗಾತ್ರ(vk_mat_mat_id_ತಳ್ಳು_constants), mul_mat_id_param_ಎಣಿಕೆ);
                sg_create({GGML_TYPE_F16, GGML_TYPE_F32, ನಿಜ, ಸುಳ್ಳು}, tc_mm, "matmul_id_f16_f32", matmul_id_f16_f32_fp32_len, matmul_id_f16_f32_fp32_ದತ್ತಾಂಶ, ಗಾತ್ರ(vk_mat_mat_id_ತಳ್ಳು_constants), mul_mat_id_param_ಎಣಿಕೆ);
                sg_create({GGML_TYPE_BF16, GGML_TYPE_BF16, ನಿಜ, ಸುಳ್ಳು}, tc_mm, "matmul_id_bf16",  matmul_id_bf16_fp32_len,    matmul_id_bf16_fp32_ದತ್ತಾಂಶ,    ಗಾತ್ರ(vk_mat_mat_id_ತಳ್ಳು_constants), mul_mat_id_param_ಎಣಿಕೆ);
                ಗಾಗಿ (ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ type : non_lut_quant_types) {
                    sg_create_quant({type, GGML_TYPE_F32, ನಿಜ, ಸುಳ್ಳು}, tc_mmqid, "matmul_id_quant_f32", matmul_id_quant_f32_fp32_len, matmul_id_quant_f32_fp32_ದತ್ತಾಂಶ, ಗಾತ್ರ(vk_mat_mat_id_ತಳ್ಳು_constants), mul_mat_id_param_ಎಣಿಕೆ);
                }
        #ವ್ಯಾಖ್ಯಾನಿಸು X_SG_ID_FP32(TYPE, tstr) \
                sg_create({TYPE, GGML_TYPE_F32, ನಿಜ, ಸುಳ್ಳು}, tc_mmqid, "matmul_id_" #tstr "_f32", matmul_id_##tstr##_f32_fp32_len, matmul_id_##tstr##_f32_fp32_data, ಗಾತ್ರ(vk_mat_mat_id_ತಳ್ಳು_constants), mul_mat_id_param_ಎಣಿಕೆ);
                FOR_EACH_LUT_TYPE(X_SG_ID_FP32)
#ವ್ಯಾಖ್ಯಾನ_ರದ್ದು X_SG_ID_FP32
            }
        }
    }
#ವ್ಯಾಖ್ಯಾನ_ರದ್ದು FOR_EACH_LUT_TYPE
#ವ್ಯಾಖ್ಯಾನ_ರದ್ದು FOR_EACH_LUT_TYPE_NONFP4
#ವ್ಯಾಖ್ಯಾನ_ರದ್ದು FOR_EACH_LUT_FP4_TYPE
    // BF16 fallback for coopmat devices without bf16 coopmat support
    ಆದರೆ ((device->coopmat2 || device->coopmat_support)
#ಆದರೆ defined(GGML_VULKAN_BFLOAT16_GLSLC_SUPPORT)
        && !device->coopmat_bf16_support
#ಆದರೆ_ಅಂತ್ಯ
        ) {
        ಸ್ಥಿರಾಂಕ uint32_t s_warptile_wm_bf16 = device->subgroup_ಗಾತ್ರ == 8 ? 8 : 32;
        std::vector<vk_tile_config> tc_bf16_fb = {
            {{ subgroup_ಗಾತ್ರ_32, 32, 32, 16, s_warptile_wm_bf16, 32, 2, 2, 2, 1, subgroup_ಗಾತ್ರ_8 }, {32, 32, 1}, s_align},
            {{ 128, 64, 64, 16, mm_warp_8, 32, 2, 4, 2, 1, mm_warp_8 }, {64, 64, 1}, m_align},
            {{ 128, 128, 128, 16, mm_warp_8 * 2, 64, 2, 4, 4, 1, mm_warp_8 }, {128, 128, 1}, l_align},
        };
        ಸ್ವಯಂಚಾಲಿತ tc_bf16_filtered = filter_tc(tc_bf16_fb, GGML_TYPE_BF16, ಸುಳ್ಳು);
        ಸ್ವಯಂಚಾಲಿತ tc_bf16_id_filtered = filter_tc(tc_bf16_fb, GGML_TYPE_BF16, ನಿಜ);
        spec_fn_t bf16_spec = [&](ಸ್ಥಿರಾಂಕ std::vector<uint32_t>& wt, ತಾರ್ಕಿಕ a) { ಹಿಂತಿರುಗಿ ggml_vk_mul_mm_spec(wt, a); };
        ಆದರೆ (!tc_bf16_filtered.ಖಾಲಿಗಳು()) {
            create_mm_pipelines({GGML_TYPE_BF16, GGML_TYPE_BF16, ಸುಳ್ಳು, ಸುಳ್ಳು}, tc_bf16_filtered, "matmul_bf16", matmul_bf16_fp32_len, matmul_bf16_fp32_ದತ್ತಾಂಶ, ಗಾತ್ರ(vk_mat_mat_ತಳ್ಳು_constants), 3, bf16_spec);
        }
        ಆದರೆ (!tc_bf16_id_filtered.ಖಾಲಿಗಳು()) {
            create_mm_pipelines({GGML_TYPE_BF16, GGML_TYPE_BF16, ನಿಜ, ಸುಳ್ಳು}, tc_bf16_id_filtered, "matmul_id_bf16", matmul_id_bf16_fp32_len, matmul_id_bf16_fp32_ದತ್ತಾಂಶ, ಗಾತ್ರ(vk_mat_mat_id_ತಳ್ಳು_constants), mul_mat_id_param_ಎಣಿಕೆ, bf16_spec);
        }
    }

    // Set up tile selector functions
    ಆದರೆ (device->coopmat2) {
        device->matmul_tile_selector = [](uint32_t m, uint32_t n, uint32_t /*k*/, uint32_t shader_core_ಎಣಿಕೆ,
                                          ಸ್ಥಿರಾಂಕ std::vector<vk_matmul_pipeline_pair>& configs) -> uint32_t {
            ಆದರೆ (configs.size() <= 1) ಹಿಂತಿರುಗಿ 0;
            uint32_t ಕೊನೆಯ = (uint32_t)configs.size() - 1;
            ಆದರೆ (configs.size() == 2) {
                uint32_t crossover = configs[0].unaligned->wg_denoms[1];
                ಹಿಂತಿರುಗಿ (n > crossover) ? 1 : 0;
            }
            // 3+ configs: s=0, m=1, l=2
            ಸ್ಥಿರಾಂಕ uint32_t tiles_l = CEIL_DIV(m, configs[ಕೊನೆಯ].unaligned->wg_denoms[0]) * CEIL_DIV(n, configs[ಕೊನೆಯ].unaligned->wg_denoms[1]);
            ಸ್ಥಿರಾಂಕ uint32_t tiles_m = CEIL_DIV(m, configs[1].unaligned->wg_denoms[0]) * CEIL_DIV(n, configs[1].unaligned->wg_denoms[1]);
            uint32_t crossover_large = configs[1].unaligned->wg_denoms[1];
            ತಾರ್ಕಿಕ prefer_large = tiles_m > shader_core_ಎಣಿಕೆ || tiles_l > shader_core_ಎಣಿಕೆ ||
                                (tiles_l <= shader_core_ಎಣಿಕೆ / 3 && tiles_m > shader_core_ಎಣಿಕೆ / 2);
            ಆದರೆ (n > crossover_large && prefer_large) ಹಿಂತಿರುಗಿ ಕೊನೆಯ;
            uint32_t crossover_medium_m = configs[0].unaligned->wg_denoms[0];
            uint32_t crossover_medium_n = configs[0].unaligned->wg_denoms[1];
            ಆದರೆ (m > crossover_medium_m && n > crossover_medium_n) ಹಿಂತಿರುಗಿ 1;
            ಹಿಂತಿರುಗಿ 0;
        };
        device->matmul_id_tile_selector = [](uint32_t /*m*/, uint32_t n, uint32_t /*k*/, uint32_t /*shader_core_count*/,
                                             ಸ್ಥಿರಾಂಕ std::vector<vk_matmul_pipeline_pair>& configs) -> uint32_t {
            ಆದರೆ (configs.size() <= 1) ಹಿಂತಿರುಗಿ 0;
            uint32_t ಕೊನೆಯ = (uint32_t)configs.size() - 1;
            ಆದರೆ (configs.size() == 2) {
                uint32_t crossover = configs[0].unaligned->wg_denoms[1];
                ಹಿಂತಿರುಗಿ (n > crossover) ? 1 : 0;
            }
            uint32_t crossover_large = configs[1].unaligned->wg_denoms[1];
            ಆದರೆ (n > crossover_large) ಹಿಂತಿರುಗಿ ಕೊನೆಯ;
            uint32_t crossover_medium = configs[0].unaligned->wg_denoms[1];
            ಆದರೆ (n > crossover_medium) ಹಿಂತಿರುಗಿ 1;
            ಹಿಂತಿರುಗಿ 0;
        };
    } ಇಲ್ಲದಿದ್ದರೆ {
        device->matmul_tile_selector = [](uint32_t m, uint32_t n, uint32_t /*k*/, uint32_t /*shader_core_count*/,
                                          ಸ್ಥಿರಾಂಕ std::vector<vk_matmul_pipeline_pair>& configs) -> uint32_t {
            ಆದರೆ (configs.size() <= 1) ಹಿಂತಿರುಗಿ 0;
            ಆದರೆ (m <= 32 || n <= 32) ಹಿಂತಿರುಗಿ 0;
            ಆದರೆ (configs.size() == 2) ಹಿಂತಿರುಗಿ 1;
            ಆದರೆ (m <= 64 || n <= 64) ಹಿಂತಿರುಗಿ 1;
            ಹಿಂತಿರುಗಿ (uint32_t)configs.size() - 1;
        };
        device->matmul_id_tile_selector = device->matmul_tile_selector;
    }

    // mul mat vec

    // the number of rows computed per shader depends on GPU model and quant
    uint32_t rm_stdq = 1;
    uint32_t rm_kq = 2;
    uint32_t rm_stdq_int = 1;
    uint32_t rm_kq_int = 1;
    ಸ್ವಯಂಚಾಲಿತ ಸ್ಥಿರಾಂಕ &rm_iq_int = [](uint32_t i) { ಹಿಂತಿರುಗಿ i == 0 ? 8u : 4u; };
    ಆದರೆ (device->vendor_id == VK_VENDOR_ID_AMD) {
        ಆದರೆ (device->architecture == AMD_GCN) {
            rm_stdq = 2;
            rm_kq = 4;
            rm_stdq_int = 4;
        }
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (device->vendor_id == VK_VENDOR_ID_INTEL) {
        rm_stdq = 2;
        rm_stdq_int = 2;
    }
    // RDNA3/4: above four columns, static 4 rows for all types bench faster than the default
    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ is_rdna3_or_4 = device->vendor_id == VK_VENDOR_ID_AMD && (device->architecture == AMD_RDNA3 || device->architecture == AMD_RDNA4);
    ಸ್ವಯಂಚಾಲಿತ ಸ್ಥಿರಾಂಕ &rm_int_n = [&](uint32_t ಅಡ್ಡಸಾಲುಗಳು, uint32_t i) { ಹಿಂತಿರುಗಿ (is_rdna3_or_4 && i >= 4) ? 4u : ಅಡ್ಡಸಾಲುಗಳು; };
    // RDNA3/4 and NVIDIA except pre-Turing: use 4 rows for MUL_MAT_ID MMVQ.
    ಸ್ವಯಂಚಾಲಿತ ಸ್ಥಿರಾಂಕ &rm_id = [&](uint32_t ಅಡ್ಡಸಾಲುಗಳು) {
        ಆದರೆ (device->vendor_id == VK_VENDOR_ID_NVIDIA &&
            device->architecture != vk_device_architecture::NVIDIA_PRE_TURING) {
            ಹಿಂತಿರುಗಿ 4u;
        }
        ಹಿಂತಿರುಗಿ is_rdna3_or_4 ? 4u : ಅಡ್ಡಸಾಲುಗಳು;
    };
    uint32_t rm_iq = 2 * rm_kq;

    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ use_subgroups = device->subgroup_arithmetic;
    // The Imagination proprietary compiler rejects the subgroup-only dequant mul_mat_vec
    // shaders that require a subgroup size >= 16; fall back to shared-memory reduction.
    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ is_imagination_proprietary =
        device->driver_id == vk::DriverId::eImaginationProprietary;
    // Ensure a subgroup size >= 16 is available
    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ use_subgroups16 = use_subgroups && subgroup_min_ಗಾತ್ರ_16 && !is_imagination_proprietary;

    ಸ್ಥಿರಾಂಕ uint32_t subgroup_ಗಾತ್ರ = (device->vendor_id == VK_VENDOR_ID_INTEL && device->subgroup_ಗಾತ್ರ_control && device->subgroup_min_ಗಾತ್ರ <= 16 && device->subgroup_max_ಗಾತ್ರ >= 16) ? 16 : device->subgroup_ಗಾತ್ರ;
    ಸ್ಥಿರಾಂಕ uint32_t subgroup_size16 = std::max(subgroup_ಗಾತ್ರ, 16u);

    ಸ್ಥಿರಾಂಕ uint32_t force_subgroup_ಗಾತ್ರ = use_subgroups ? subgroup_ಗಾತ್ರ : 0;
    ಸ್ಥಿರಾಂಕ uint32_t force_subgroup_size16 = use_subgroups16 ? subgroup_size16 : 0;
    ಸ್ಥಿರ constexpr uint32_t mul_mat_vec_ಅಂಕೆ_bindings = 5;
    ಸ್ಥಿರ constexpr uint32_t mul_mat_vec_id_ಅಂಕೆ_bindings = 6;

#ಆದರೆ defined(GGML_VULKAN_FLOAT_E2M1_GLSLC_SUPPORT) && defined(GGML_VULKAN_FLOAT_E4M3_GLSLC_SUPPORT)
#ವ್ಯಾಖ್ಯಾನಿಸು OCP_DMMV_LEN(NAME, REDUC)  (device->ocp_fp4 ? NAME ## _ocp_len[REDUC]  : NAME ## _len[REDUC])
#ವ್ಯಾಖ್ಯಾನಿಸು OCP_DMMV_ದತ್ತಾಂಶ(NAME, REDUC) (device->ocp_fp4 ? NAME ## ocp_ದತ್ತಾಂಶ[REDUC] : NAME ## ದತ್ತಾಂಶ_2[REDUC])
#ಇಲ್ಲದಿದ್ದರೆ
#ವ್ಯಾಖ್ಯಾನಿಸು OCP_DMMV_LEN(NAME, REDUC)  NAME ## _len[REDUC]
#ವ್ಯಾಖ್ಯಾನಿಸು OCP_DMMV_ದತ್ತಾಂಶ(NAME, REDUC) NAME ## ದತ್ತಾಂಶ_2[REDUC]
#ಆದರೆ_ಅಂತ್ಯ

    ಗಾಗಿ (uint32_t w = 0; w < DMMV_WG_ಗಾತ್ರ_ಎಣಿಕೆ; ++w) {
        ಸ್ಥಿರಾಂಕ uint32_t wg_ಗಾತ್ರ_subgroup   = (w == DMMV_WG_ಗಾತ್ರ_SUBGROUP) ? subgroup_ಗಾತ್ರ : (subgroup_ಗಾತ್ರ * 4);
        ಸ್ಥಿರಾಂಕ uint32_t wg_ಗಾತ್ರ_subgroup16 = (w == DMMV_WG_ಗಾತ್ರ_SUBGROUP) ? subgroup_size16 : (subgroup_size16 * 4);

        ಸ್ಥಿರಾಂಕ shader_reduction_ಬಹುಲಕ reduc = (use_subgroups && w == DMMV_WG_ಗಾತ್ರ_SUBGROUP) ? SHADER_REDUCTION_ಬಹುಲಕ_SUBGROUP :
                                            (use_subgroups && w == DMMV_WG_ಗಾತ್ರ_LARGE) ? SHADER_REDUCTION_ಬಹುಲಕ_HYBRID :
                                            SHADER_REDUCTION_ಬಹುಲಕ_SHMEM;

        ಸ್ಥಿರಾಂಕ shader_reduction_ಬಹುಲಕ reduc16 = (use_subgroups16 && w == DMMV_WG_ಗಾತ್ರ_SUBGROUP) ? SHADER_REDUCTION_ಬಹುಲಕ_SUBGROUP :
                                              (use_subgroups16 && w == DMMV_WG_ಗಾತ್ರ_LARGE) ? SHADER_REDUCTION_ಬಹುಲಕ_HYBRID :
                                              SHADER_REDUCTION_ಬಹುಲಕ_SHMEM;

        ಗಾಗಿ (uint32_t i = 0; i < mul_mat_vec_max_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ; ++i) {
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_F32 ][i], "mul_mat_vec_f32_f32_f32",  arr_dmmv_f32_f32_f32_len[reduc],  arr_dmmv_f32_f32_f32_ದತ್ತಾಂಶ[reduc],  "main", mul_mat_vec_ಅಂಕೆ_bindings, ಗಾತ್ರ(vk_mat_vec_ತಳ್ಳು_constants), {1, 1, 1}, {wg_ಗಾತ್ರ_subgroup, 1, i+1}, 1, ಸುಳ್ಳು, use_subgroups, force_subgroup_ಗಾತ್ರ);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_F16 ][i], "mul_mat_vec_f16_f32_f32",  arr_dmmv_f16_f32_f32_len[reduc],  arr_dmmv_f16_f32_f32_ದತ್ತಾಂಶ[reduc],  "main", mul_mat_vec_ಅಂಕೆ_bindings, ಗಾತ್ರ(vk_mat_vec_ತಳ್ಳು_constants), {2, 1, 1}, {wg_ಗಾತ್ರ_subgroup, 2, i+1}, 1, ಸುಳ್ಳು, use_subgroups, force_subgroup_ಗಾತ್ರ);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_BF16][i], "mul_mat_vec_bf16_f32_f32", arr_dmmv_bf16_f32_f32_len[reduc], arr_dmmv_bf16_f32_f32_ದತ್ತಾಂಶ[reduc], "main", mul_mat_vec_ಅಂಕೆ_bindings, ಗಾತ್ರ(vk_mat_vec_ತಳ್ಳು_constants), {2, 1, 1}, {wg_ಗಾತ್ರ_subgroup, 2, i+1}, 1, ಸುಳ್ಳು, use_subgroups, force_subgroup_ಗಾತ್ರ);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_Q1_0][i], "mul_mat_vec_q1_0_f32_f32", arr_dmmv_q1_0_f32_f32_len[reduc], arr_dmmv_q1_0_f32_f32_ದತ್ತಾಂಶ[reduc], "main", mul_mat_vec_ಅಂಕೆ_bindings, ಗಾತ್ರ(vk_mat_vec_ತಳ್ಳು_constants), {2*rm_stdq, 1, 1}, {wg_ಗಾತ್ರ_subgroup, 2*rm_stdq, i+1}, 1, ನಿಜ, use_subgroups, force_subgroup_ಗಾತ್ರ);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_Q2_0][i], "mul_mat_vec_q2_0_f32_f32", arr_dmmv_q2_0_f32_f32_len[reduc], arr_dmmv_q2_0_f32_f32_ದತ್ತಾಂಶ[reduc], "main", mul_mat_vec_ಅಂಕೆ_bindings, ಗಾತ್ರ(vk_mat_vec_ತಳ್ಳು_constants), {2*rm_stdq, 1, 1}, {wg_ಗಾತ್ರ_subgroup, 2*rm_stdq, i+1}, 1, ನಿಜ, use_subgroups, force_subgroup_ಗಾತ್ರ);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_Q4_0][i], "mul_mat_vec_q4_0_f32_f32", arr_dmmv_q4_0_f32_f32_len[reduc], arr_dmmv_q4_0_f32_f32_ದತ್ತಾಂಶ[reduc], "main", mul_mat_vec_ಅಂಕೆ_bindings, ಗಾತ್ರ(vk_mat_vec_ತಳ್ಳು_constants), {2*rm_stdq, 1, 1}, {wg_ಗಾತ್ರ_subgroup, 2*rm_stdq, i+1}, 1, ನಿಜ, use_subgroups, force_subgroup_ಗಾತ್ರ);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_Q4_1][i], "mul_mat_vec_q4_1_f32_f32", arr_dmmv_q4_1_f32_f32_len[reduc], arr_dmmv_q4_1_f32_f32_ದತ್ತಾಂಶ[reduc], "main", mul_mat_vec_ಅಂಕೆ_bindings, ಗಾತ್ರ(vk_mat_vec_ತಳ್ಳು_constants), {2*rm_stdq, 1, 1}, {wg_ಗಾತ್ರ_subgroup, 2*rm_stdq, i+1}, 1, ನಿಜ, use_subgroups, force_subgroup_ಗಾತ್ರ);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_Q5_0][i], "mul_mat_vec_q5_0_f32_f32", arr_dmmv_q5_0_f32_f32_len[reduc], arr_dmmv_q5_0_f32_f32_ದತ್ತಾಂಶ[reduc], "main", mul_mat_vec_ಅಂಕೆ_bindings, ಗಾತ್ರ(vk_mat_vec_ತಳ್ಳು_constants), {2*rm_stdq, 1, 1}, {wg_ಗಾತ್ರ_subgroup, 2*rm_stdq, i+1}, 1, ನಿಜ, use_subgroups, force_subgroup_ಗಾತ್ರ);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_Q5_1][i], "mul_mat_vec_q5_1_f32_f32", arr_dmmv_q5_1_f32_f32_len[reduc], arr_dmmv_q5_1_f32_f32_ದತ್ತಾಂಶ[reduc], "main", mul_mat_vec_ಅಂಕೆ_bindings, ಗಾತ್ರ(vk_mat_vec_ತಳ್ಳು_constants), {2*rm_stdq, 1, 1}, {wg_ಗಾತ್ರ_subgroup, 2*rm_stdq, i+1}, 1, ನಿಜ, use_subgroups, force_subgroup_ಗಾತ್ರ);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_Q8_0][i], "mul_mat_vec_q8_0_f32_f32", arr_dmmv_q8_0_f32_f32_len[reduc], arr_dmmv_q8_0_f32_f32_ದತ್ತಾಂಶ[reduc], "main", mul_mat_vec_ಅಂಕೆ_bindings, ಗಾತ್ರ(vk_mat_vec_ತಳ್ಳು_constants), {1*rm_stdq, 1, 1}, {wg_ಗಾತ್ರ_subgroup, 1*rm_stdq, i+1}, 1, ನಿಜ, use_subgroups, force_subgroup_ಗಾತ್ರ);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_Q2_K][i], "mul_mat_vec_q2_k_f32_f32", arr_dmmv_q2_k_f32_f32_len[reduc16], arr_dmmv_q2_k_f32_f32_ದತ್ತಾಂಶ[reduc16], "main", mul_mat_vec_ಅಂಕೆ_bindings, ಗಾತ್ರ(vk_mat_vec_ತಳ್ಳು_constants), {rm_kq, 1, 1}, {wg_ಗಾತ್ರ_subgroup16, rm_kq, i+1}, 1, ನಿಜ, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_Q3_K][i], "mul_mat_vec_q3_k_f32_f32", arr_dmmv_q3_k_f32_f32_len[reduc16], arr_dmmv_q3_k_f32_f32_ದತ್ತಾಂಶ[reduc16], "main", mul_mat_vec_ಅಂಕೆ_bindings, ಗಾತ್ರ(vk_mat_vec_ತಳ್ಳು_constants), {rm_kq, 1, 1}, {wg_ಗಾತ್ರ_subgroup16, rm_kq, i+1}, 1, ನಿಜ, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_Q4_K][i], "mul_mat_vec_q4_k_f32_f32", arr_dmmv_q4_k_f32_f32_len[reduc16], arr_dmmv_q4_k_f32_f32_ದತ್ತಾಂಶ[reduc16], "main", mul_mat_vec_ಅಂಕೆ_bindings, ಗಾತ್ರ(vk_mat_vec_ತಳ್ಳು_constants), {rm_kq, 1, 1}, {wg_ಗಾತ್ರ_subgroup16, rm_kq, i+1}, 1, ನಿಜ, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_Q5_K][i], "mul_mat_vec_q5_k_f32_f32", arr_dmmv_q5_k_f32_f32_len[reduc16], arr_dmmv_q5_k_f32_f32_ದತ್ತಾಂಶ[reduc16], "main", mul_mat_vec_ಅಂಕೆ_bindings, ಗಾತ್ರ(vk_mat_vec_ತಳ್ಳು_constants), {rm_kq, 1, 1}, {wg_ಗಾತ್ರ_subgroup16, rm_kq, i+1}, 1, ನಿಜ, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_Q6_K][i], "mul_mat_vec_q6_k_f32_f32", arr_dmmv_q6_k_f32_f32_len[reduc16], arr_dmmv_q6_k_f32_f32_ದತ್ತಾಂಶ[reduc16], "main", mul_mat_vec_ಅಂಕೆ_bindings, ಗಾತ್ರ(vk_mat_vec_ತಳ್ಳು_constants), {rm_kq, 1, 1}, {wg_ಗಾತ್ರ_subgroup16, rm_kq, i+1}, 1, ನಿಜ, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_TQ1_0][i], "mul_mat_vec_tq1_0_f32_f32", arr_dmmv_tq1_0_f32_f32_len[reduc16], arr_dmmv_tq1_0_f32_f32_ದತ್ತಾಂಶ[reduc16], "main", mul_mat_vec_ಅಂಕೆ_bindings, ಗಾತ್ರ(vk_mat_vec_ತಳ್ಳು_constants), {rm_kq, 1, 1}, {wg_ಗಾತ್ರ_subgroup16, rm_kq, i+1}, 1, ನಿಜ, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_TQ2_0][i], "mul_mat_vec_tq2_0_f32_f32", arr_dmmv_tq2_0_f32_f32_len[reduc16], arr_dmmv_tq2_0_f32_f32_ದತ್ತಾಂಶ[reduc16], "main", mul_mat_vec_ಅಂಕೆ_bindings, ಗಾತ್ರ(vk_mat_vec_ತಳ್ಳು_constants), {rm_kq, 1, 1}, {wg_ಗಾತ್ರ_subgroup16, rm_kq, i+1}, 1, ನಿಜ, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_IQ1_S][i],   "mul_mat_vec_iq1_s_f32_f32",   arr_dmmv_iq1_s_f32_f32_len[reduc16],   arr_dmmv_iq1_s_f32_f32_ದತ್ತಾಂಶ[reduc16],   "main", mul_mat_vec_ಅಂಕೆ_bindings, ಗಾತ್ರ(vk_mat_vec_ತಳ್ಳು_constants), {rm_iq, 1, 1}, {wg_ಗಾತ್ರ_subgroup16, rm_iq, i+1}, 1, ನಿಜ, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_IQ1_M][i],   "mul_mat_vec_iq1_m_f32_f32",   arr_dmmv_iq1_m_f32_f32_len[reduc16],   arr_dmmv_iq1_m_f32_f32_ದತ್ತಾಂಶ[reduc16],   "main", mul_mat_vec_ಅಂಕೆ_bindings, ಗಾತ್ರ(vk_mat_vec_ತಳ್ಳು_constants), {rm_iq, 1, 1}, {wg_ಗಾತ್ರ_subgroup16, rm_iq, i+1}, 1, ನಿಜ, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_IQ2_XXS][i], "mul_mat_vec_iq2_xxs_f32_f32", arr_dmmv_iq2_xxs_f32_f32_len[reduc16], arr_dmmv_iq2_xxs_f32_f32_ದತ್ತಾಂಶ[reduc16], "main", mul_mat_vec_ಅಂಕೆ_bindings, ಗಾತ್ರ(vk_mat_vec_ತಳ್ಳು_constants), {rm_iq, 1, 1}, {wg_ಗಾತ್ರ_subgroup16, rm_iq, i+1}, 1, ನಿಜ, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_IQ2_XS][i],  "mul_mat_vec_iq2_xs_f32_f32",  arr_dmmv_iq2_xs_f32_f32_len[reduc16],  arr_dmmv_iq2_xs_f32_f32_ದತ್ತಾಂಶ[reduc16],  "main", mul_mat_vec_ಅಂಕೆ_bindings, ಗಾತ್ರ(vk_mat_vec_ತಳ್ಳು_constants), {rm_iq, 1, 1}, {wg_ಗಾತ್ರ_subgroup16, rm_iq, i+1}, 1, ನಿಜ, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_IQ2_S][i],   "mul_mat_vec_iq2_s_f32_f32",   arr_dmmv_iq2_s_f32_f32_len[reduc16],   arr_dmmv_iq2_s_f32_f32_ದತ್ತಾಂಶ[reduc16],   "main", mul_mat_vec_ಅಂಕೆ_bindings, ಗಾತ್ರ(vk_mat_vec_ತಳ್ಳು_constants), {rm_iq, 1, 1}, {wg_ಗಾತ್ರ_subgroup16, rm_iq, i+1}, 1, ನಿಜ, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_IQ3_XXS][i], "mul_mat_vec_iq3_xxs_f32_f32", arr_dmmv_iq3_xxs_f32_f32_len[reduc16], arr_dmmv_iq3_xxs_f32_f32_ದತ್ತಾಂಶ[reduc16], "main", mul_mat_vec_ಅಂಕೆ_bindings, ಗಾತ್ರ(vk_mat_vec_ತಳ್ಳು_constants), {rm_iq, 1, 1}, {wg_ಗಾತ್ರ_subgroup16, rm_iq, i+1}, 1, ನಿಜ, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_IQ3_S][i],   "mul_mat_vec_iq3_s_f32_f32",   arr_dmmv_iq3_s_f32_f32_len[reduc16],   arr_dmmv_iq3_s_f32_f32_ದತ್ತಾಂಶ[reduc16],   "main", mul_mat_vec_ಅಂಕೆ_bindings, ಗಾತ್ರ(vk_mat_vec_ತಳ್ಳು_constants), {rm_iq, 1, 1}, {wg_ಗಾತ್ರ_subgroup16, rm_iq, i+1}, 1, ನಿಜ, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_IQ4_XS][i],  "mul_mat_vec_iq4_xs_f32_f32",  arr_dmmv_iq4_xs_f32_f32_len[reduc16],  arr_dmmv_iq4_xs_f32_f32_ದತ್ತಾಂಶ[reduc16],  "main", mul_mat_vec_ಅಂಕೆ_bindings, ಗಾತ್ರ(vk_mat_vec_ತಳ್ಳು_constants), {rm_iq, 1, 1}, {wg_ಗಾತ್ರ_subgroup16, rm_iq, i+1}, 1, ನಿಜ, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_IQ4_NL][i],  "mul_mat_vec_iq4_nl_f32_f32",  arr_dmmv_iq4_nl_f32_f32_len[reduc16],  arr_dmmv_iq4_nl_f32_f32_ದತ್ತಾಂಶ[reduc16],  "main", mul_mat_vec_ಅಂಕೆ_bindings, ಗಾತ್ರ(vk_mat_vec_ತಳ್ಳು_constants), {rm_iq, 1, 1}, {wg_ಗಾತ್ರ_subgroup16, rm_iq, i+1}, 1, ನಿಜ, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_MXFP4][i],   "mul_mat_vec_mxfp4_f32_f32",   OCP_DMMV_LEN(arr_dmmv_mxfp4_f32_f32, reduc16), OCP_DMMV_ದತ್ತಾಂಶ(arr_dmmv_mxfp4_f32_f32, reduc16), "main", mul_mat_vec_ಅಂಕೆ_bindings, ಗಾತ್ರ(vk_mat_vec_ತಳ್ಳು_constants), {rm_iq, 1, 1}, {wg_ಗಾತ್ರ_subgroup16, rm_iq, i+1}, 1, ನಿಜ, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_NVFP4][i],   "mul_mat_vec_nvfp4_f32_f32",   OCP_DMMV_LEN(arr_dmmv_nvfp4_f32_f32, reduc16), OCP_DMMV_ದತ್ತಾಂಶ(arr_dmmv_nvfp4_f32_f32, reduc16), "main", mul_mat_vec_ಅಂಕೆ_bindings, ಗಾತ್ರ(vk_mat_vec_ತಳ್ಳು_constants), {rm_iq, 1, 1}, {wg_ಗಾತ್ರ_subgroup16, rm_iq, i+1}, 1, ನಿಜ, use_subgroups16, force_subgroup_size16);

            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_F32 ][i], "mul_mat_vec_f32_f16_f32",  arr_dmmv_f32_f16_f32_len[reduc],  arr_dmmv_f32_f16_f32_ದತ್ತಾಂಶ[reduc],  "main", mul_mat_vec_ಅಂಕೆ_bindings, ಗಾತ್ರ(vk_mat_vec_ತಳ್ಳು_constants), {1, 1, 1}, {wg_ಗಾತ್ರ_subgroup, 1, i+1}, 1, ಸುಳ್ಳು, use_subgroups, force_subgroup_ಗಾತ್ರ);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_F16 ][i], "mul_mat_vec_f16_f16_f32",  arr_dmmv_f16_f16_f32_len[reduc],  arr_dmmv_f16_f16_f32_ದತ್ತಾಂಶ[reduc],  "main", mul_mat_vec_ಅಂಕೆ_bindings, ಗಾತ್ರ(vk_mat_vec_ತಳ್ಳು_constants), {2, 1, 1}, {wg_ಗಾತ್ರ_subgroup, 2, i+1}, 1, ಸುಳ್ಳು, use_subgroups, force_subgroup_ಗಾತ್ರ);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_BF16][i], "mul_mat_vec_bf16_f16_f32", arr_dmmv_bf16_f16_f32_len[reduc], arr_dmmv_bf16_f16_f32_ದತ್ತಾಂಶ[reduc], "main", mul_mat_vec_ಅಂಕೆ_bindings, ಗಾತ್ರ(vk_mat_vec_ತಳ್ಳು_constants), {2, 1, 1}, {wg_ಗಾತ್ರ_subgroup, 2, i+1}, 1, ಸುಳ್ಳು, use_subgroups, force_subgroup_ಗಾತ್ರ);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_Q1_0][i], "mul_mat_vec_q1_0_f16_f32", arr_dmmv_q1_0_f16_f32_len[reduc], arr_dmmv_q1_0_f16_f32_ದತ್ತಾಂಶ[reduc], "main", mul_mat_vec_ಅಂಕೆ_bindings, ಗಾತ್ರ(vk_mat_vec_ತಳ್ಳು_constants), {2*rm_stdq, 1, 1}, {wg_ಗಾತ್ರ_subgroup, 2*rm_stdq, i+1}, 1, ನಿಜ, use_subgroups, force_subgroup_ಗಾತ್ರ);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_Q2_0][i], "mul_mat_vec_q2_0_f16_f32", arr_dmmv_q2_0_f16_f32_len[reduc], arr_dmmv_q2_0_f16_f32_ದತ್ತಾಂಶ[reduc], "main", mul_mat_vec_ಅಂಕೆ_bindings, ಗಾತ್ರ(vk_mat_vec_ತಳ್ಳು_constants), {2*rm_stdq, 1, 1}, {wg_ಗಾತ್ರ_subgroup, 2*rm_stdq, i+1}, 1, ನಿಜ, use_subgroups, force_subgroup_ಗಾತ್ರ);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_Q4_0][i], "mul_mat_vec_q4_0_f16_f32", arr_dmmv_q4_0_f16_f32_len[reduc], arr_dmmv_q4_0_f16_f32_ದತ್ತಾಂಶ[reduc], "main", mul_mat_vec_ಅಂಕೆ_bindings, ಗಾತ್ರ(vk_mat_vec_ತಳ್ಳು_constants), {2*rm_stdq, 1, 1}, {wg_ಗಾತ್ರ_subgroup, 2*rm_stdq, i+1}, 1, ನಿಜ, use_subgroups, force_subgroup_ಗಾತ್ರ);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_Q4_1][i], "mul_mat_vec_q4_1_f16_f32", arr_dmmv_q4_1_f16_f32_len[reduc], arr_dmmv_q4_1_f16_f32_ದತ್ತಾಂಶ[reduc], "main", mul_mat_vec_ಅಂಕೆ_bindings, ಗಾತ್ರ(vk_mat_vec_ತಳ್ಳು_constants), {2*rm_stdq, 1, 1}, {wg_ಗಾತ್ರ_subgroup, 2*rm_stdq, i+1}, 1, ನಿಜ, use_subgroups, force_subgroup_ಗಾತ್ರ);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_Q5_0][i], "mul_mat_vec_q5_0_f16_f32", arr_dmmv_q5_0_f16_f32_len[reduc], arr_dmmv_q5_0_f16_f32_ದತ್ತಾಂಶ[reduc], "main", mul_mat_vec_ಅಂಕೆ_bindings, ಗಾತ್ರ(vk_mat_vec_ತಳ್ಳು_constants), {2*rm_stdq, 1, 1}, {wg_ಗಾತ್ರ_subgroup, 2*rm_stdq, i+1}, 1, ನಿಜ, use_subgroups, force_subgroup_ಗಾತ್ರ);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_Q5_1][i], "mul_mat_vec_q5_1_f16_f32", arr_dmmv_q5_1_f16_f32_len[reduc], arr_dmmv_q5_1_f16_f32_ದತ್ತಾಂಶ[reduc], "main", mul_mat_vec_ಅಂಕೆ_bindings, ಗಾತ್ರ(vk_mat_vec_ತಳ್ಳು_constants), {2*rm_stdq, 1, 1}, {wg_ಗಾತ್ರ_subgroup, 2*rm_stdq, i+1}, 1, ನಿಜ, use_subgroups, force_subgroup_ಗಾತ್ರ);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_Q8_0][i], "mul_mat_vec_q8_0_f16_f32", arr_dmmv_q8_0_f16_f32_len[reduc], arr_dmmv_q8_0_f16_f32_ದತ್ತಾಂಶ[reduc], "main", mul_mat_vec_ಅಂಕೆ_bindings, ಗಾತ್ರ(vk_mat_vec_ತಳ್ಳು_constants), {1*rm_stdq, 1, 1}, {wg_ಗಾತ್ರ_subgroup, 1*rm_stdq, i+1}, 1, ನಿಜ, use_subgroups, force_subgroup_ಗಾತ್ರ);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_Q2_K][i], "mul_mat_vec_q2_k_f16_f32", arr_dmmv_q2_k_f16_f32_len[reduc16], arr_dmmv_q2_k_f16_f32_ದತ್ತಾಂಶ[reduc16], "main", mul_mat_vec_ಅಂಕೆ_bindings, ಗಾತ್ರ(vk_mat_vec_ತಳ್ಳು_constants), {rm_kq, 1, 1}, {wg_ಗಾತ್ರ_subgroup16, rm_kq, i+1}, 1, ನಿಜ, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_Q3_K][i], "mul_mat_vec_q3_k_f16_f32", arr_dmmv_q3_k_f16_f32_len[reduc16], arr_dmmv_q3_k_f16_f32_ದತ್ತಾಂಶ[reduc16], "main", mul_mat_vec_ಅಂಕೆ_bindings, ಗಾತ್ರ(vk_mat_vec_ತಳ್ಳು_constants), {rm_kq, 1, 1}, {wg_ಗಾತ್ರ_subgroup16, rm_kq, i+1}, 1, ನಿಜ, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_Q4_K][i], "mul_mat_vec_q4_k_f16_f32", arr_dmmv_q4_k_f16_f32_len[reduc16], arr_dmmv_q4_k_f16_f32_ದತ್ತಾಂಶ[reduc16], "main", mul_mat_vec_ಅಂಕೆ_bindings, ಗಾತ್ರ(vk_mat_vec_ತಳ್ಳು_constants), {rm_kq, 1, 1}, {wg_ಗಾತ್ರ_subgroup16, rm_kq, i+1}, 1, ನಿಜ, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_Q5_K][i], "mul_mat_vec_q5_k_f16_f32", arr_dmmv_q5_k_f16_f32_len[reduc16], arr_dmmv_q5_k_f16_f32_ದತ್ತಾಂಶ[reduc16], "main", mul_mat_vec_ಅಂಕೆ_bindings, ಗಾತ್ರ(vk_mat_vec_ತಳ್ಳು_constants), {rm_kq, 1, 1}, {wg_ಗಾತ್ರ_subgroup16, rm_kq, i+1}, 1, ನಿಜ, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_Q6_K][i], "mul_mat_vec_q6_k_f16_f32", arr_dmmv_q6_k_f16_f32_len[reduc16], arr_dmmv_q6_k_f16_f32_ದತ್ತಾಂಶ[reduc16], "main", mul_mat_vec_ಅಂಕೆ_bindings, ಗಾತ್ರ(vk_mat_vec_ತಳ್ಳು_constants), {rm_kq, 1, 1}, {wg_ಗಾತ್ರ_subgroup16, rm_kq, i+1}, 1, ನಿಜ, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_TQ1_0][i], "mul_mat_vec_tq1_0_f16_f32", arr_dmmv_tq1_0_f16_f32_len[reduc16], arr_dmmv_tq1_0_f16_f32_ದತ್ತಾಂಶ[reduc16], "main", mul_mat_vec_ಅಂಕೆ_bindings, ಗಾತ್ರ(vk_mat_vec_ತಳ್ಳು_constants), {rm_kq, 1, 1}, {wg_ಗಾತ್ರ_subgroup16, rm_kq, i+1}, 1, ನಿಜ, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_TQ2_0][i], "mul_mat_vec_tq2_0_f16_f32", arr_dmmv_tq2_0_f16_f32_len[reduc16], arr_dmmv_tq2_0_f16_f32_ದತ್ತಾಂಶ[reduc16], "main", mul_mat_vec_ಅಂಕೆ_bindings, ಗಾತ್ರ(vk_mat_vec_ತಳ್ಳು_constants), {rm_kq, 1, 1}, {wg_ಗಾತ್ರ_subgroup16, rm_kq, i+1}, 1, ನಿಜ, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_IQ1_S][i],   "mul_mat_vec_iq1_s_f16_f32",   arr_dmmv_iq1_s_f16_f32_len[reduc16],   arr_dmmv_iq1_s_f16_f32_ದತ್ತಾಂಶ[reduc16],   "main", mul_mat_vec_ಅಂಕೆ_bindings, ಗಾತ್ರ(vk_mat_vec_ತಳ್ಳು_constants), {rm_iq, 1, 1}, {wg_ಗಾತ್ರ_subgroup16, rm_iq, i+1}, 1, ನಿಜ, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_IQ1_M][i],   "mul_mat_vec_iq1_m_f16_f32",   arr_dmmv_iq1_m_f16_f32_len[reduc16],   arr_dmmv_iq1_m_f16_f32_ದತ್ತಾಂಶ[reduc16],   "main", mul_mat_vec_ಅಂಕೆ_bindings, ಗಾತ್ರ(vk_mat_vec_ತಳ್ಳು_constants), {rm_iq, 1, 1}, {wg_ಗಾತ್ರ_subgroup16, rm_iq, i+1}, 1, ನಿಜ, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_IQ2_XXS][i], "mul_mat_vec_iq2_xxs_f16_f32", arr_dmmv_iq2_xxs_f16_f32_len[reduc16], arr_dmmv_iq2_xxs_f16_f32_ದತ್ತಾಂಶ[reduc16], "main", mul_mat_vec_ಅಂಕೆ_bindings, ಗಾತ್ರ(vk_mat_vec_ತಳ್ಳು_constants), {rm_iq, 1, 1}, {wg_ಗಾತ್ರ_subgroup16, rm_iq, i+1}, 1, ನಿಜ, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_IQ2_XS][i],  "mul_mat_vec_iq2_xs_f16_f32",  arr_dmmv_iq2_xs_f16_f32_len[reduc16],  arr_dmmv_iq2_xs_f16_f32_ದತ್ತಾಂಶ[reduc16],  "main", mul_mat_vec_ಅಂಕೆ_bindings, ಗಾತ್ರ(vk_mat_vec_ತಳ್ಳು_constants), {rm_iq, 1, 1}, {wg_ಗಾತ್ರ_subgroup16, rm_iq, i+1}, 1, ನಿಜ, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_IQ2_S][i],   "mul_mat_vec_iq2_s_f16_f32",   arr_dmmv_iq2_s_f16_f32_len[reduc16],   arr_dmmv_iq2_s_f16_f32_ದತ್ತಾಂಶ[reduc16],   "main", mul_mat_vec_ಅಂಕೆ_bindings, ಗಾತ್ರ(vk_mat_vec_ತಳ್ಳು_constants), {rm_iq, 1, 1}, {wg_ಗಾತ್ರ_subgroup16, rm_iq, i+1}, 1, ನಿಜ, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_IQ3_XXS][i], "mul_mat_vec_iq3_xxs_f16_f32", arr_dmmv_iq3_xxs_f16_f32_len[reduc16], arr_dmmv_iq3_xxs_f16_f32_ದತ್ತಾಂಶ[reduc16], "main", mul_mat_vec_ಅಂಕೆ_bindings, ಗಾತ್ರ(vk_mat_vec_ತಳ್ಳು_constants), {rm_iq, 1, 1}, {wg_ಗಾತ್ರ_subgroup16, rm_iq, i+1}, 1, ನಿಜ, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_IQ3_S][i],   "mul_mat_vec_iq3_s_f16_f32",   arr_dmmv_iq3_s_f16_f32_len[reduc16],   arr_dmmv_iq3_s_f16_f32_ದತ್ತಾಂಶ[reduc16],   "main", mul_mat_vec_ಅಂಕೆ_bindings, ಗಾತ್ರ(vk_mat_vec_ತಳ್ಳು_constants), {rm_iq, 1, 1}, {wg_ಗಾತ್ರ_subgroup16, rm_iq, i+1}, 1, ನಿಜ, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_IQ4_XS][i],  "mul_mat_vec_iq4_xs_f16_f32",  arr_dmmv_iq4_xs_f16_f32_len[reduc16],  arr_dmmv_iq4_xs_f16_f32_ದತ್ತಾಂಶ[reduc16],  "main", mul_mat_vec_ಅಂಕೆ_bindings, ಗಾತ್ರ(vk_mat_vec_ತಳ್ಳು_constants), {rm_iq, 1, 1}, {wg_ಗಾತ್ರ_subgroup16, rm_iq, i+1}, 1, ನಿಜ, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_IQ4_NL][i],  "mul_mat_vec_iq4_nl_f16_f32",  arr_dmmv_iq4_nl_f16_f32_len[reduc16],  arr_dmmv_iq4_nl_f16_f32_ದತ್ತಾಂಶ[reduc16],  "main", mul_mat_vec_ಅಂಕೆ_bindings, ಗಾತ್ರ(vk_mat_vec_ತಳ್ಳು_constants), {rm_iq, 1, 1}, {wg_ಗಾತ್ರ_subgroup16, rm_iq, i+1}, 1, ನಿಜ, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_MXFP4][i],   "mul_mat_vec_mxfp4_f16_f32",   OCP_DMMV_LEN(arr_dmmv_mxfp4_f16_f32, reduc16), OCP_DMMV_ದತ್ತಾಂಶ(arr_dmmv_mxfp4_f16_f32, reduc16), "main", mul_mat_vec_ಅಂಕೆ_bindings, ಗಾತ್ರ(vk_mat_vec_ತಳ್ಳು_constants), {rm_iq, 1, 1}, {wg_ಗಾತ್ರ_subgroup16, rm_iq, i+1}, 1, ನಿಜ, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_NVFP4][i],   "mul_mat_vec_nvfp4_f16_f32",   OCP_DMMV_LEN(arr_dmmv_nvfp4_f16_f32, reduc16), OCP_DMMV_ದತ್ತಾಂಶ(arr_dmmv_nvfp4_f16_f32, reduc16), "main", mul_mat_vec_ಅಂಕೆ_bindings, ಗಾತ್ರ(vk_mat_vec_ತಳ್ಳು_constants), {rm_iq, 1, 1}, {wg_ಗಾತ್ರ_subgroup16, rm_iq, i+1}, 1, ನಿಜ, use_subgroups16, force_subgroup_size16);

#ಆದರೆ defined(GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT)
            ಆದರೆ (device->integer_dot_ಗುಣಲಬ್ಧ) {
                ಸ್ಥಿರಾಂಕ uint32_t subgroup_ಗಾತ್ರ_int = (device->vendor_id == VK_VENDOR_ID_INTEL && device->subgroup_ಗಾತ್ರ_control) ? device->subgroup_min_ಗಾತ್ರ : device->subgroup_ಗಾತ್ರ;
                ಸ್ಥಿರಾಂಕ uint32_t wg_ಗಾತ್ರ_subgroup_int = (w == DMMV_WG_ಗಾತ್ರ_SUBGROUP) ? subgroup_ಗಾತ್ರ_int : (subgroup_ಗಾತ್ರ_int * 4);

                ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_q8_1_f32[w][GGML_TYPE_Q2_0][i], "mul_mat_vec_q2_0_q8_1_f32", arr_dmmv_q2_0_q8_1_f32_len[reduc], arr_dmmv_q2_0_q8_1_f32_ದತ್ತಾಂಶ[reduc], "main", mul_mat_vec_ಅಂಕೆ_bindings, ಗಾತ್ರ(vk_mat_vec_ತಳ್ಳು_constants), {rm_int_n(2*rm_kq_int, i), 1, 1}, {wg_ಗಾತ್ರ_subgroup_int, rm_int_n(2*rm_kq_int, i), i+1}, 1, ನಿಜ, use_subgroups, subgroup_ಗಾತ್ರ_int);
                ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_q8_1_f32[w][GGML_TYPE_Q4_0][i], "mul_mat_vec_q4_0_q8_1_f32", arr_dmmv_q4_0_q8_1_f32_len[reduc], arr_dmmv_q4_0_q8_1_f32_ದತ್ತಾಂಶ[reduc], "main", mul_mat_vec_ಅಂಕೆ_bindings, ಗಾತ್ರ(vk_mat_vec_ತಳ್ಳು_constants), {rm_int_n(1*rm_stdq_int, i), 1, 1}, {wg_ಗಾತ್ರ_subgroup_int, rm_int_n(1*rm_stdq_int, i), i+1}, 1, ನಿಜ, use_subgroups, subgroup_ಗಾತ್ರ_int);
                ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_q8_1_f32[w][GGML_TYPE_Q4_1][i], "mul_mat_vec_q4_1_q8_1_f32", arr_dmmv_q4_1_q8_1_f32_len[reduc], arr_dmmv_q4_1_q8_1_f32_ದತ್ತಾಂಶ[reduc], "main", mul_mat_vec_ಅಂಕೆ_bindings, ಗಾತ್ರ(vk_mat_vec_ತಳ್ಳು_constants), {rm_int_n(1*rm_stdq_int, i), 1, 1}, {wg_ಗಾತ್ರ_subgroup_int, rm_int_n(1*rm_stdq_int, i), i+1}, 1, ನಿಜ, use_subgroups, subgroup_ಗಾತ್ರ_int);
                ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_q8_1_f32[w][GGML_TYPE_Q5_0][i], "mul_mat_vec_q5_0_q8_1_f32", arr_dmmv_q5_0_q8_1_f32_len[reduc], arr_dmmv_q5_0_q8_1_f32_ದತ್ತಾಂಶ[reduc], "main", mul_mat_vec_ಅಂಕೆ_bindings, ಗಾತ್ರ(vk_mat_vec_ತಳ್ಳು_constants), {rm_int_n(1*rm_stdq_int, i), 1, 1}, {wg_ಗಾತ್ರ_subgroup_int, rm_int_n(1*rm_stdq_int, i), i+1}, 1, ನಿಜ, use_subgroups, subgroup_ಗಾತ್ರ_int);
                ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_q8_1_f32[w][GGML_TYPE_Q5_1][i], "mul_mat_vec_q5_1_q8_1_f32", arr_dmmv_q5_1_q8_1_f32_len[reduc], arr_dmmv_q5_1_q8_1_f32_ದತ್ತಾಂಶ[reduc], "main", mul_mat_vec_ಅಂಕೆ_bindings, ಗಾತ್ರ(vk_mat_vec_ತಳ್ಳು_constants), {rm_int_n(1*rm_stdq_int, i), 1, 1}, {wg_ಗಾತ್ರ_subgroup_int, rm_int_n(1*rm_stdq_int, i), i+1}, 1, ನಿಜ, use_subgroups, subgroup_ಗಾತ್ರ_int);
                ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_q8_1_f32[w][GGML_TYPE_Q8_0][i], "mul_mat_vec_q8_0_q8_1_f32", arr_dmmv_q8_0_q8_1_f32_len[reduc], arr_dmmv_q8_0_q8_1_f32_ದತ್ತಾಂಶ[reduc], "main", mul_mat_vec_ಅಂಕೆ_bindings, ಗಾತ್ರ(vk_mat_vec_ತಳ್ಳು_constants), {rm_int_n(1*rm_stdq_int, i), 1, 1}, {wg_ಗಾತ್ರ_subgroup_int, rm_int_n(1*rm_stdq_int, i), i+1}, 1, ನಿಜ, use_subgroups, subgroup_ಗಾತ್ರ_int);

                ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_q8_1_f32[w][GGML_TYPE_MXFP4][i], "mul_mat_vec_mxfp4_q8_1_f32", arr_dmmv_mxfp4_q8_1_f32_len[reduc], arr_dmmv_mxfp4_q8_1_f32_ದತ್ತಾಂಶ[reduc], "main", mul_mat_vec_ಅಂಕೆ_bindings, ಗಾತ್ರ(vk_mat_vec_ತಳ್ಳು_constants), {rm_int_n(2*rm_stdq_int, i), 1, 1}, {wg_ಗಾತ್ರ_subgroup_int, rm_int_n(2*rm_stdq_int, i), i+1}, 1, ನಿಜ, use_subgroups, subgroup_ಗಾತ್ರ_int);

                ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_q8_1_f32[w][GGML_TYPE_Q2_K][i], "mul_mat_vec_q2_k_q8_1_f32", arr_dmmv_q2_k_q8_1_f32_len[reduc], arr_dmmv_q2_k_q8_1_f32_ದತ್ತಾಂಶ[reduc], "main", mul_mat_vec_ಅಂಕೆ_bindings, ಗಾತ್ರ(vk_mat_vec_ತಳ್ಳು_constants), {rm_int_n(2*rm_kq_int, i), 1, 1}, {wg_ಗಾತ್ರ_subgroup_int, rm_int_n(2*rm_kq_int, i), i+1}, 1, ನಿಜ, use_subgroups, subgroup_ಗಾತ್ರ_int);
                ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_q8_1_f32[w][GGML_TYPE_Q3_K][i], "mul_mat_vec_q3_k_q8_1_f32", arr_dmmv_q3_k_q8_1_f32_len[reduc], arr_dmmv_q3_k_q8_1_f32_ದತ್ತಾಂಶ[reduc], "main", mul_mat_vec_ಅಂಕೆ_bindings, ಗಾತ್ರ(vk_mat_vec_ತಳ್ಳು_constants), {rm_int_n(1*rm_kq_int, i), 1, 1}, {wg_ಗಾತ್ರ_subgroup_int, rm_int_n(1*rm_kq_int, i), i+1}, 1, ನಿಜ, use_subgroups, subgroup_ಗಾತ್ರ_int);
                ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_q8_1_f32[w][GGML_TYPE_Q4_K][i], "mul_mat_vec_q4_k_q8_1_f32", arr_dmmv_q4_k_q8_1_f32_len[reduc], arr_dmmv_q4_k_q8_1_f32_ದತ್ತಾಂಶ[reduc], "main", mul_mat_vec_ಅಂಕೆ_bindings, ಗಾತ್ರ(vk_mat_vec_ತಳ್ಳು_constants), {rm_int_n(1*rm_kq_int, i), 1, 1}, {wg_ಗಾತ್ರ_subgroup_int, rm_int_n(1*rm_kq_int, i), i+1}, 1, ನಿಜ, use_subgroups, subgroup_ಗಾತ್ರ_int);
                ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_q8_1_f32[w][GGML_TYPE_Q5_K][i], "mul_mat_vec_q5_k_q8_1_f32", arr_dmmv_q5_k_q8_1_f32_len[reduc], arr_dmmv_q5_k_q8_1_f32_ದತ್ತಾಂಶ[reduc], "main", mul_mat_vec_ಅಂಕೆ_bindings, ಗಾತ್ರ(vk_mat_vec_ತಳ್ಳು_constants), {rm_int_n(1*rm_kq_int, i), 1, 1}, {wg_ಗಾತ್ರ_subgroup_int, rm_int_n(1*rm_kq_int, i), i+1}, 1, ನಿಜ, use_subgroups, subgroup_ಗಾತ್ರ_int);
                ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_q8_1_f32[w][GGML_TYPE_Q6_K][i], "mul_mat_vec_q6_k_q8_1_f32", arr_dmmv_q6_k_q8_1_f32_len[reduc], arr_dmmv_q6_k_q8_1_f32_ದತ್ತಾಂಶ[reduc], "main", mul_mat_vec_ಅಂಕೆ_bindings, ಗಾತ್ರ(vk_mat_vec_ತಳ್ಳು_constants), {rm_int_n(1*rm_kq_int, i), 1, 1}, {wg_ಗಾತ್ರ_subgroup_int, rm_int_n(1*rm_kq_int, i), i+1}, 1, ನಿಜ, use_subgroups, subgroup_ಗಾತ್ರ_int);

                ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_q8_1_f32[w][GGML_TYPE_IQ1_S][i], "mul_mat_vec_iq1_s_q8_1_f32", arr_dmmv_iq1_s_q8_1_f32_len[reduc], arr_dmmv_iq1_s_q8_1_f32_ದತ್ತಾಂಶ[reduc], "main", mul_mat_vec_ಅಂಕೆ_bindings, ಗಾತ್ರ(vk_mat_vec_ತಳ್ಳು_constants), {1*rm_iq_int(i), 1, 1}, {wg_ಗಾತ್ರ_subgroup_int, 1*rm_iq_int(i), i+1}, 1, ನಿಜ, use_subgroups, subgroup_ಗಾತ್ರ_int);
                ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_q8_1_f32[w][GGML_TYPE_IQ1_M][i], "mul_mat_vec_iq1_m_q8_1_f32", arr_dmmv_iq1_m_q8_1_f32_len[reduc], arr_dmmv_iq1_m_q8_1_f32_ದತ್ತಾಂಶ[reduc], "main", mul_mat_vec_ಅಂಕೆ_bindings, ಗಾತ್ರ(vk_mat_vec_ತಳ್ಳು_constants), {1*rm_iq_int(i), 1, 1}, {wg_ಗಾತ್ರ_subgroup_int, 1*rm_iq_int(i), i+1}, 1, ನಿಜ, use_subgroups, subgroup_ಗಾತ್ರ_int);
                ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_q8_1_f32[w][GGML_TYPE_IQ4_XS][i], "mul_mat_vec_iq4_xs_q8_1_f32", arr_dmmv_iq4_xs_q8_1_f32_len[reduc], arr_dmmv_iq4_xs_q8_1_f32_ದತ್ತಾಂಶ[reduc], "main", mul_mat_vec_ಅಂಕೆ_bindings, ಗಾತ್ರ(vk_mat_vec_ತಳ್ಳು_constants), {rm_int_n(1*rm_stdq_int, i), 1, 1}, {wg_ಗಾತ್ರ_subgroup_int, rm_int_n(1*rm_stdq_int, i), i+1}, 1, ನಿಜ, use_subgroups, subgroup_ಗಾತ್ರ_int);

            }
#ಆದರೆ_ಅಂತ್ಯ // GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT
        }

        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_F32 ], "mul_mat_vec_id_f32_f32",        arr_dmmv_id_f32_f32_f32_len[reduc],     arr_dmmv_id_f32_f32_f32_ದತ್ತಾಂಶ[reduc],     "main", mul_mat_vec_id_ಅಂಕೆ_bindings, ಗಾತ್ರ(vk_mat_vec_id_ತಳ್ಳು_constants), {1, 1, 1}, {wg_ಗಾತ್ರ_subgroup, 1}, 1, ಸುಳ್ಳು, use_subgroups, force_subgroup_ಗಾತ್ರ);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_F16 ], "mul_mat_vec_id_f16_f32",        arr_dmmv_id_f16_f32_f32_len[reduc],     arr_dmmv_id_f16_f32_f32_ದತ್ತಾಂಶ[reduc],     "main", mul_mat_vec_id_ಅಂಕೆ_bindings, ಗಾತ್ರ(vk_mat_vec_id_ತಳ್ಳು_constants), {2, 1, 1}, {wg_ಗಾತ್ರ_subgroup, 2}, 1, ಸುಳ್ಳು, use_subgroups, force_subgroup_ಗಾತ್ರ);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_BF16], "mul_mat_vec_id_bf16_f32",       arr_dmmv_id_bf16_f32_f32_len[reduc],    arr_dmmv_id_bf16_f32_f32_ದತ್ತಾಂಶ[reduc],    "main", mul_mat_vec_id_ಅಂಕೆ_bindings, ಗಾತ್ರ(vk_mat_vec_id_ತಳ್ಳು_constants), {2, 1, 1}, {wg_ಗಾತ್ರ_subgroup, 2}, 1, ಸುಳ್ಳು, use_subgroups, force_subgroup_ಗಾತ್ರ);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_Q1_0], "mul_mat_vec_id_q1_0_f32",       arr_dmmv_id_q1_0_f32_f32_len[reduc],    arr_dmmv_id_q1_0_f32_f32_ದತ್ತಾಂಶ[reduc],    "main", mul_mat_vec_id_ಅಂಕೆ_bindings, ಗಾತ್ರ(vk_mat_vec_id_ತಳ್ಳು_constants), {2*rm_stdq, 1, 1}, {wg_ಗಾತ್ರ_subgroup, 2*rm_stdq}, 1, ನಿಜ, use_subgroups, force_subgroup_ಗಾತ್ರ);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_Q2_0], "mul_mat_vec_id_q2_0_f32",       arr_dmmv_id_q2_0_f32_f32_len[reduc],    arr_dmmv_id_q2_0_f32_f32_ದತ್ತಾಂಶ[reduc],    "main", mul_mat_vec_id_ಅಂಕೆ_bindings, ಗಾತ್ರ(vk_mat_vec_id_ತಳ್ಳು_constants), {2*rm_stdq, 1, 1}, {wg_ಗಾತ್ರ_subgroup, 2*rm_stdq}, 1, ನಿಜ, use_subgroups, force_subgroup_ಗಾತ್ರ);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_Q4_0], "mul_mat_vec_id_q4_0_f32",       arr_dmmv_id_q4_0_f32_f32_len[reduc],    arr_dmmv_id_q4_0_f32_f32_ದತ್ತಾಂಶ[reduc],    "main", mul_mat_vec_id_ಅಂಕೆ_bindings, ಗಾತ್ರ(vk_mat_vec_id_ತಳ್ಳು_constants), {2*rm_stdq, 1, 1}, {wg_ಗಾತ್ರ_subgroup, 2*rm_stdq}, 1, ನಿಜ, use_subgroups, force_subgroup_ಗಾತ್ರ);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_Q4_1], "mul_mat_vec_id_q4_1_f32",       arr_dmmv_id_q4_1_f32_f32_len[reduc],    arr_dmmv_id_q4_1_f32_f32_ದತ್ತಾಂಶ[reduc],    "main", mul_mat_vec_id_ಅಂಕೆ_bindings, ಗಾತ್ರ(vk_mat_vec_id_ತಳ್ಳು_constants), {2*rm_stdq, 1, 1}, {wg_ಗಾತ್ರ_subgroup, 2*rm_stdq}, 1, ನಿಜ, use_subgroups, force_subgroup_ಗಾತ್ರ);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_Q5_0], "mul_mat_vec_id_q5_0_f32",       arr_dmmv_id_q5_0_f32_f32_len[reduc],    arr_dmmv_id_q5_0_f32_f32_ದತ್ತಾಂಶ[reduc],    "main", mul_mat_vec_id_ಅಂಕೆ_bindings, ಗಾತ್ರ(vk_mat_vec_id_ತಳ್ಳು_constants), {2*rm_stdq, 1, 1}, {wg_ಗಾತ್ರ_subgroup, 2*rm_stdq}, 1, ನಿಜ, use_subgroups, force_subgroup_ಗಾತ್ರ);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_Q5_1], "mul_mat_vec_id_q5_1_f32",       arr_dmmv_id_q5_1_f32_f32_len[reduc],    arr_dmmv_id_q5_1_f32_f32_ದತ್ತಾಂಶ[reduc],    "main", mul_mat_vec_id_ಅಂಕೆ_bindings, ಗಾತ್ರ(vk_mat_vec_id_ತಳ್ಳು_constants), {2*rm_stdq, 1, 1}, {wg_ಗಾತ್ರ_subgroup, 2*rm_stdq}, 1, ನಿಜ, use_subgroups, force_subgroup_ಗಾತ್ರ);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_Q8_0], "mul_mat_vec_id_q8_0_f32",       arr_dmmv_id_q8_0_f32_f32_len[reduc],    arr_dmmv_id_q8_0_f32_f32_ದತ್ತಾಂಶ[reduc],    "main", mul_mat_vec_id_ಅಂಕೆ_bindings, ಗಾತ್ರ(vk_mat_vec_id_ತಳ್ಳು_constants), {1*rm_stdq, 1, 1}, {wg_ಗಾತ್ರ_subgroup, 1*rm_stdq}, 1, ನಿಜ, use_subgroups, force_subgroup_ಗಾತ್ರ);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_Q2_K], "mul_mat_vec_id_q2_k_f32",       arr_dmmv_id_q2_k_f32_f32_len[reduc16],    arr_dmmv_id_q2_k_f32_f32_ದತ್ತಾಂಶ[reduc16],    "main", mul_mat_vec_id_ಅಂಕೆ_bindings, ಗಾತ್ರ(vk_mat_vec_id_ತಳ್ಳು_constants), {rm_kq, 1, 1}, {wg_ಗಾತ್ರ_subgroup16, rm_kq}, 1, ನಿಜ, use_subgroups16, force_subgroup_size16);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_Q3_K], "mul_mat_vec_id_q3_k_f32",       arr_dmmv_id_q3_k_f32_f32_len[reduc16],    arr_dmmv_id_q3_k_f32_f32_ದತ್ತಾಂಶ[reduc16],    "main", mul_mat_vec_id_ಅಂಕೆ_bindings, ಗಾತ್ರ(vk_mat_vec_id_ತಳ್ಳು_constants), {rm_kq, 1, 1}, {wg_ಗಾತ್ರ_subgroup16, rm_kq}, 1, ನಿಜ, use_subgroups16, force_subgroup_size16);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_Q4_K], "mul_mat_vec_id_q4_k_f32",       arr_dmmv_id_q4_k_f32_f32_len[reduc16],    arr_dmmv_id_q4_k_f32_f32_ದತ್ತಾಂಶ[reduc16],    "main", mul_mat_vec_id_ಅಂಕೆ_bindings, ಗಾತ್ರ(vk_mat_vec_id_ತಳ್ಳು_constants), {rm_kq, 1, 1}, {wg_ಗಾತ್ರ_subgroup16, rm_kq}, 1, ನಿಜ, use_subgroups16, force_subgroup_size16);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_Q5_K], "mul_mat_vec_id_q5_k_f32",       arr_dmmv_id_q5_k_f32_f32_len[reduc16],    arr_dmmv_id_q5_k_f32_f32_ದತ್ತಾಂಶ[reduc16],    "main", mul_mat_vec_id_ಅಂಕೆ_bindings, ಗಾತ್ರ(vk_mat_vec_id_ತಳ್ಳು_constants), {rm_kq, 1, 1}, {wg_ಗಾತ್ರ_subgroup16, rm_kq}, 1, ನಿಜ, use_subgroups16, force_subgroup_size16);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_Q6_K], "mul_mat_vec_id_q6_k_f32",       arr_dmmv_id_q6_k_f32_f32_len[reduc16],    arr_dmmv_id_q6_k_f32_f32_ದತ್ತಾಂಶ[reduc16],    "main", mul_mat_vec_id_ಅಂಕೆ_bindings, ಗಾತ್ರ(vk_mat_vec_id_ತಳ್ಳು_constants), {rm_kq, 1, 1}, {wg_ಗಾತ್ರ_subgroup16, rm_kq}, 1, ನಿಜ, use_subgroups16, force_subgroup_size16);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_TQ1_0], "mul_mat_vec_id_tq1_0_f32",     arr_dmmv_id_tq1_0_f32_f32_len[reduc16],   arr_dmmv_id_tq1_0_f32_f32_ದತ್ತಾಂಶ[reduc16],   "main", mul_mat_vec_id_ಅಂಕೆ_bindings, ಗಾತ್ರ(vk_mat_vec_id_ತಳ್ಳು_constants), {rm_kq, 1, 1}, {wg_ಗಾತ್ರ_subgroup16, rm_kq}, 1, ನಿಜ, use_subgroups16, force_subgroup_size16);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_TQ2_0], "mul_mat_vec_id_tq2_0_f32",     arr_dmmv_id_tq2_0_f32_f32_len[reduc16],   arr_dmmv_id_tq2_0_f32_f32_ದತ್ತಾಂಶ[reduc16],   "main", mul_mat_vec_id_ಅಂಕೆ_bindings, ಗಾತ್ರ(vk_mat_vec_id_ತಳ್ಳು_constants), {rm_kq, 1, 1}, {wg_ಗಾತ್ರ_subgroup16, rm_kq}, 1, ನಿಜ, use_subgroups16, force_subgroup_size16);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_IQ1_S],   "mul_mat_vec_id_iq1_s_f32",   arr_dmmv_id_iq1_s_f32_f32_len[reduc16],   arr_dmmv_id_iq1_s_f32_f32_ದತ್ತಾಂಶ[reduc16],   "main", mul_mat_vec_id_ಅಂಕೆ_bindings, ಗಾತ್ರ(vk_mat_vec_id_ತಳ್ಳು_constants), {rm_iq, 1, 1}, {wg_ಗಾತ್ರ_subgroup16, rm_iq}, 1, ನಿಜ, use_subgroups16, force_subgroup_size16);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_IQ1_M],   "mul_mat_vec_id_iq1_m_f32",   arr_dmmv_id_iq1_m_f32_f32_len[reduc16],   arr_dmmv_id_iq1_m_f32_f32_ದತ್ತಾಂಶ[reduc16],   "main", mul_mat_vec_id_ಅಂಕೆ_bindings, ಗಾತ್ರ(vk_mat_vec_id_ತಳ್ಳು_constants), {rm_iq, 1, 1}, {wg_ಗಾತ್ರ_subgroup16, rm_iq}, 1, ನಿಜ, use_subgroups16, force_subgroup_size16);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_IQ2_XXS], "mul_mat_vec_id_iq2_xxs_f32", arr_dmmv_id_iq2_xxs_f32_f32_len[reduc16], arr_dmmv_id_iq2_xxs_f32_f32_ದತ್ತಾಂಶ[reduc16], "main", mul_mat_vec_id_ಅಂಕೆ_bindings, ಗಾತ್ರ(vk_mat_vec_id_ತಳ್ಳು_constants), {rm_iq, 1, 1}, {wg_ಗಾತ್ರ_subgroup16, rm_iq}, 1, ನಿಜ, use_subgroups16, force_subgroup_size16);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_IQ2_XS],  "mul_mat_vec_id_iq2_xs_f32",  arr_dmmv_id_iq2_xs_f32_f32_len[reduc16],  arr_dmmv_id_iq2_xs_f32_f32_ದತ್ತಾಂಶ[reduc16],  "main", mul_mat_vec_id_ಅಂಕೆ_bindings, ಗಾತ್ರ(vk_mat_vec_id_ತಳ್ಳು_constants), {rm_iq, 1, 1}, {wg_ಗಾತ್ರ_subgroup16, rm_iq}, 1, ನಿಜ, use_subgroups16, force_subgroup_size16);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_IQ2_S],   "mul_mat_vec_id_iq2_s_f32",   arr_dmmv_id_iq2_s_f32_f32_len[reduc16],   arr_dmmv_id_iq2_s_f32_f32_ದತ್ತಾಂಶ[reduc16],   "main", mul_mat_vec_id_ಅಂಕೆ_bindings, ಗಾತ್ರ(vk_mat_vec_id_ತಳ್ಳು_constants), {rm_iq, 1, 1}, {wg_ಗಾತ್ರ_subgroup16, rm_iq}, 1, ನಿಜ, use_subgroups16, force_subgroup_size16);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_IQ3_XXS], "mul_mat_vec_id_iq3_xxs_f32", arr_dmmv_id_iq3_xxs_f32_f32_len[reduc16], arr_dmmv_id_iq3_xxs_f32_f32_ದತ್ತಾಂಶ[reduc16], "main", mul_mat_vec_id_ಅಂಕೆ_bindings, ಗಾತ್ರ(vk_mat_vec_id_ತಳ್ಳು_constants), {rm_iq, 1, 1}, {wg_ಗಾತ್ರ_subgroup16, rm_iq}, 1, ನಿಜ, use_subgroups16, force_subgroup_size16);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_IQ3_S],   "mul_mat_vec_id_iq3_s_f32",   arr_dmmv_id_iq3_s_f32_f32_len[reduc16],   arr_dmmv_id_iq3_s_f32_f32_ದತ್ತಾಂಶ[reduc16],   "main", mul_mat_vec_id_ಅಂಕೆ_bindings, ಗಾತ್ರ(vk_mat_vec_id_ತಳ್ಳು_constants), {rm_iq, 1, 1}, {wg_ಗಾತ್ರ_subgroup16, rm_iq}, 1, ನಿಜ, use_subgroups16, force_subgroup_size16);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_IQ4_XS],  "mul_mat_vec_id_iq4_xs_f32",  arr_dmmv_id_iq4_xs_f32_f32_len[reduc16],  arr_dmmv_id_iq4_xs_f32_f32_ದತ್ತಾಂಶ[reduc16],  "main", mul_mat_vec_id_ಅಂಕೆ_bindings, ಗಾತ್ರ(vk_mat_vec_id_ತಳ್ಳು_constants), {rm_iq, 1, 1}, {wg_ಗಾತ್ರ_subgroup16, rm_iq}, 1, ನಿಜ, use_subgroups16, force_subgroup_size16);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_IQ4_NL],  "mul_mat_vec_id_iq4_nl_f32",  arr_dmmv_id_iq4_nl_f32_f32_len[reduc16],  arr_dmmv_id_iq4_nl_f32_f32_ದತ್ತಾಂಶ[reduc16],  "main", mul_mat_vec_id_ಅಂಕೆ_bindings, ಗಾತ್ರ(vk_mat_vec_id_ತಳ್ಳು_constants), {rm_iq, 1, 1}, {wg_ಗಾತ್ರ_subgroup16, rm_iq}, 1, ನಿಜ, use_subgroups16, force_subgroup_size16);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_MXFP4],   "mul_mat_vec_id_mxfp4_f32",   OCP_DMMV_LEN(arr_dmmv_id_mxfp4_f32_f32, reduc16), OCP_DMMV_ದತ್ತಾಂಶ(arr_dmmv_id_mxfp4_f32_f32, reduc16), "main", mul_mat_vec_id_ಅಂಕೆ_bindings, ಗಾತ್ರ(vk_mat_vec_id_ತಳ್ಳು_constants), {rm_iq, 1, 1}, {wg_ಗಾತ್ರ_subgroup16, rm_iq}, 1, ನಿಜ, use_subgroups16, force_subgroup_size16);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_NVFP4],   "mul_mat_vec_id_nvfp4_f32",   OCP_DMMV_LEN(arr_dmmv_id_nvfp4_f32_f32, reduc16), OCP_DMMV_ದತ್ತಾಂಶ(arr_dmmv_id_nvfp4_f32_f32, reduc16), "main", mul_mat_vec_id_ಅಂಕೆ_bindings, ಗಾತ್ರ(vk_mat_vec_id_ತಳ್ಳು_constants), {rm_iq, 1, 1}, {wg_ಗಾತ್ರ_subgroup16, rm_iq}, 1, ನಿಜ, use_subgroups16, force_subgroup_size16);

#ಆದರೆ defined(GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT)
        ಆದರೆ (device->integer_dot_ಗುಣಲಬ್ಧ) {
            ಸ್ಥಿರಾಂಕ uint32_t subgroup_ಗಾತ್ರ_int = (device->vendor_id == VK_VENDOR_ID_INTEL && device->subgroup_ಗಾತ್ರ_control) ? device->subgroup_min_ಗಾತ್ರ : device->subgroup_ಗಾತ್ರ;
            ಸ್ಥಿರಾಂಕ uint32_t wg_ಗಾತ್ರ_subgroup_int = (w == DMMV_WG_ಗಾತ್ರ_SUBGROUP) ? subgroup_ಗಾತ್ರ_int : (subgroup_ಗಾತ್ರ_int * 4);

            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_q8_1_f32[w][GGML_TYPE_Q2_0], "mul_mat_vec_id_q2_0_q8_1_f32", arr_dmmv_id_q2_0_q8_1_f32_len[reduc], arr_dmmv_id_q2_0_q8_1_f32_ದತ್ತಾಂಶ[reduc], "main", mul_mat_vec_id_ಅಂಕೆ_bindings, ಗಾತ್ರ(vk_mat_vec_id_ತಳ್ಳು_constants), {rm_id(2*rm_kq_int), 1, 1}, {wg_ಗಾತ್ರ_subgroup_int, rm_id(2*rm_kq_int)}, 1, ನಿಜ, use_subgroups, subgroup_ಗಾತ್ರ_int);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_q8_1_f32[w][GGML_TYPE_Q4_0], "mul_mat_vec_id_q4_0_q8_1_f32", arr_dmmv_id_q4_0_q8_1_f32_len[reduc], arr_dmmv_id_q4_0_q8_1_f32_ದತ್ತಾಂಶ[reduc], "main", mul_mat_vec_id_ಅಂಕೆ_bindings, ಗಾತ್ರ(vk_mat_vec_id_ತಳ್ಳು_constants), {rm_id(1*rm_stdq_int), 1, 1}, {wg_ಗಾತ್ರ_subgroup_int, rm_id(1*rm_stdq_int)}, 1, ನಿಜ, use_subgroups, subgroup_ಗಾತ್ರ_int);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_q8_1_f32[w][GGML_TYPE_Q4_1], "mul_mat_vec_id_q4_1_q8_1_f32", arr_dmmv_id_q4_1_q8_1_f32_len[reduc], arr_dmmv_id_q4_1_q8_1_f32_ದತ್ತಾಂಶ[reduc], "main", mul_mat_vec_id_ಅಂಕೆ_bindings, ಗಾತ್ರ(vk_mat_vec_id_ತಳ್ಳು_constants), {rm_id(1*rm_stdq_int), 1, 1}, {wg_ಗಾತ್ರ_subgroup_int, rm_id(1*rm_stdq_int)}, 1, ನಿಜ, use_subgroups, subgroup_ಗಾತ್ರ_int);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_q8_1_f32[w][GGML_TYPE_Q5_0], "mul_mat_vec_id_q5_0_q8_1_f32", arr_dmmv_id_q5_0_q8_1_f32_len[reduc], arr_dmmv_id_q5_0_q8_1_f32_ದತ್ತಾಂಶ[reduc], "main", mul_mat_vec_id_ಅಂಕೆ_bindings, ಗಾತ್ರ(vk_mat_vec_id_ತಳ್ಳು_constants), {rm_id(1*rm_stdq_int), 1, 1}, {wg_ಗಾತ್ರ_subgroup_int, rm_id(1*rm_stdq_int)}, 1, ನಿಜ, use_subgroups, subgroup_ಗಾತ್ರ_int);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_q8_1_f32[w][GGML_TYPE_Q5_1], "mul_mat_vec_id_q5_1_q8_1_f32", arr_dmmv_id_q5_1_q8_1_f32_len[reduc], arr_dmmv_id_q5_1_q8_1_f32_ದತ್ತಾಂಶ[reduc], "main", mul_mat_vec_id_ಅಂಕೆ_bindings, ಗಾತ್ರ(vk_mat_vec_id_ತಳ್ಳು_constants), {rm_id(1*rm_stdq_int), 1, 1}, {wg_ಗಾತ್ರ_subgroup_int, rm_id(1*rm_stdq_int)}, 1, ನಿಜ, use_subgroups, subgroup_ಗಾತ್ರ_int);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_q8_1_f32[w][GGML_TYPE_Q8_0], "mul_mat_vec_id_q8_0_q8_1_f32", arr_dmmv_id_q8_0_q8_1_f32_len[reduc], arr_dmmv_id_q8_0_q8_1_f32_ದತ್ತಾಂಶ[reduc], "main", mul_mat_vec_id_ಅಂಕೆ_bindings, ಗಾತ್ರ(vk_mat_vec_id_ತಳ್ಳು_constants), {rm_id(1*rm_stdq_int), 1, 1}, {wg_ಗಾತ್ರ_subgroup_int, rm_id(1*rm_stdq_int)}, 1, ನಿಜ, use_subgroups, subgroup_ಗಾತ್ರ_int);

            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_q8_1_f32[w][GGML_TYPE_MXFP4], "mul_mat_vec_id_mxfp4_q8_1_f32", arr_dmmv_id_mxfp4_q8_1_f32_len[reduc], arr_dmmv_id_mxfp4_q8_1_f32_ದತ್ತಾಂಶ[reduc], "main", mul_mat_vec_id_ಅಂಕೆ_bindings, ಗಾತ್ರ(vk_mat_vec_id_ತಳ್ಳು_constants), {rm_id(2*rm_stdq_int), 1, 1}, {wg_ಗಾತ್ರ_subgroup_int, rm_id(2*rm_stdq_int)}, 1, ನಿಜ, use_subgroups, subgroup_ಗಾತ್ರ_int);

            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_q8_1_f32[w][GGML_TYPE_Q2_K], "mul_mat_vec_id_q2_k_q8_1_f32", arr_dmmv_id_q2_k_q8_1_f32_len[reduc], arr_dmmv_id_q2_k_q8_1_f32_ದತ್ತಾಂಶ[reduc], "main", mul_mat_vec_id_ಅಂಕೆ_bindings, ಗಾತ್ರ(vk_mat_vec_id_ತಳ್ಳು_constants), {rm_id(2*rm_kq_int), 1, 1}, {wg_ಗಾತ್ರ_subgroup_int, rm_id(2*rm_kq_int)}, 1, ನಿಜ, use_subgroups, subgroup_ಗಾತ್ರ_int);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_q8_1_f32[w][GGML_TYPE_Q3_K], "mul_mat_vec_id_q3_k_q8_1_f32", arr_dmmv_id_q3_k_q8_1_f32_len[reduc], arr_dmmv_id_q3_k_q8_1_f32_ದತ್ತಾಂಶ[reduc], "main", mul_mat_vec_id_ಅಂಕೆ_bindings, ಗಾತ್ರ(vk_mat_vec_id_ತಳ್ಳು_constants), {rm_id(1*rm_kq_int), 1, 1}, {wg_ಗಾತ್ರ_subgroup_int, rm_id(1*rm_kq_int)}, 1, ನಿಜ, use_subgroups, subgroup_ಗಾತ್ರ_int);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_q8_1_f32[w][GGML_TYPE_Q4_K], "mul_mat_vec_id_q4_k_q8_1_f32", arr_dmmv_id_q4_k_q8_1_f32_len[reduc], arr_dmmv_id_q4_k_q8_1_f32_ದತ್ತಾಂಶ[reduc], "main", mul_mat_vec_id_ಅಂಕೆ_bindings, ಗಾತ್ರ(vk_mat_vec_id_ತಳ್ಳು_constants), {rm_id(1*rm_kq_int), 1, 1}, {wg_ಗಾತ್ರ_subgroup_int, rm_id(1*rm_kq_int)}, 1, ನಿಜ, use_subgroups, subgroup_ಗಾತ್ರ_int);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_q8_1_f32[w][GGML_TYPE_Q5_K], "mul_mat_vec_id_q5_k_q8_1_f32", arr_dmmv_id_q5_k_q8_1_f32_len[reduc], arr_dmmv_id_q5_k_q8_1_f32_ದತ್ತಾಂಶ[reduc], "main", mul_mat_vec_id_ಅಂಕೆ_bindings, ಗಾತ್ರ(vk_mat_vec_id_ತಳ್ಳು_constants), {rm_id(1*rm_kq_int), 1, 1}, {wg_ಗಾತ್ರ_subgroup_int, rm_id(1*rm_kq_int)}, 1, ನಿಜ, use_subgroups, subgroup_ಗಾತ್ರ_int);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_q8_1_f32[w][GGML_TYPE_Q6_K], "mul_mat_vec_id_q6_k_q8_1_f32", arr_dmmv_id_q6_k_q8_1_f32_len[reduc], arr_dmmv_id_q6_k_q8_1_f32_ದತ್ತಾಂಶ[reduc], "main", mul_mat_vec_id_ಅಂಕೆ_bindings, ಗಾತ್ರ(vk_mat_vec_id_ತಳ್ಳು_constants), {rm_id(1*rm_kq_int), 1, 1}, {wg_ಗಾತ್ರ_subgroup_int, rm_id(1*rm_kq_int)}, 1, ನಿಜ, use_subgroups, subgroup_ಗಾತ್ರ_int);

            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_q8_1_f32[w][GGML_TYPE_IQ1_S], "mul_mat_vec_id_iq1_s_q8_1_f32", arr_dmmv_id_iq1_s_q8_1_f32_len[reduc], arr_dmmv_id_iq1_s_q8_1_f32_ದತ್ತಾಂಶ[reduc], "main", mul_mat_vec_id_ಅಂಕೆ_bindings, ಗಾತ್ರ(vk_mat_vec_id_ತಳ್ಳು_constants), {1*rm_iq_int(0), 1, 1}, {wg_ಗಾತ್ರ_subgroup_int, 1*rm_iq_int(0)}, 1, ನಿಜ, use_subgroups, subgroup_ಗಾತ್ರ_int);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_q8_1_f32[w][GGML_TYPE_IQ1_M], "mul_mat_vec_id_iq1_m_q8_1_f32", arr_dmmv_id_iq1_m_q8_1_f32_len[reduc], arr_dmmv_id_iq1_m_q8_1_f32_ದತ್ತಾಂಶ[reduc], "main", mul_mat_vec_id_ಅಂಕೆ_bindings, ಗಾತ್ರ(vk_mat_vec_id_ತಳ್ಳು_constants), {1*rm_iq_int(0), 1, 1}, {wg_ಗಾತ್ರ_subgroup_int, 1*rm_iq_int(0)}, 1, ನಿಜ, use_subgroups, subgroup_ಗಾತ್ರ_int);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_q8_1_f32[w][GGML_TYPE_IQ4_XS], "mul_mat_vec_id_iq4_xs_q8_1_f32", arr_dmmv_id_iq4_xs_q8_1_f32_len[reduc], arr_dmmv_id_iq4_xs_q8_1_f32_ದತ್ತಾಂಶ[reduc], "main", mul_mat_vec_id_ಅಂಕೆ_bindings, ಗಾತ್ರ(vk_mat_vec_id_ತಳ್ಳು_constants), {rm_id(1*rm_stdq_int), 1, 1}, {wg_ಗಾತ್ರ_subgroup_int, rm_id(1*rm_stdq_int)}, 1, ನಿಜ, use_subgroups, subgroup_ಗಾತ್ರ_int);
        }
#ಆದರೆ_ಅಂತ್ಯ // GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT
    }

#ವ್ಯಾಖ್ಯಾನ_ರದ್ದು OCP_DMMV_ದತ್ತಾಂಶ
#ವ್ಯಾಖ್ಯಾನ_ರದ್ದು OCP_DMMV_LEN

#ಆದರೆ !defined(GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT)
    GGML_ಬಳಸದ(rm_stdq_int);
    GGML_ಬಳಸದ(rm_kq_int);
    GGML_ಬಳಸದ(is_rdna3_or_4);
    GGML_ಬಳಸದ(rm_int_n);
    GGML_ಬಳಸದ(rm_id);
    GGML_ಬಳಸದ(rm_iq_int);
#ಆದರೆ_ಅಂತ್ಯ

    // dequant shaders
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_F32 ], "f32_to_f16",   dequant_f32_len,  dequant_f32_ದತ್ತಾಂಶ,  "main", 2, 5 * ಗಾತ್ರ(uint32_t), {256 * 16, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_Q1_0], "dequant_q1_0", dequant_q1_0_len, dequant_q1_0_ದತ್ತಾಂಶ, "main", 2, 5 * ಗಾತ್ರ(uint32_t), {256 * 8, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_Q2_0], "dequant_q2_0", dequant_q2_0_len, dequant_q2_0_ದತ್ತಾಂಶ, "main", 2, 5 * ಗಾತ್ರ(uint32_t), {256 * 4, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_Q4_0], "dequant_q4_0", dequant_q4_0_len, dequant_q4_0_ದತ್ತಾಂಶ, "main", 2, 5 * ಗಾತ್ರ(uint32_t), {256 * 16, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_Q4_1], "dequant_q4_1", dequant_q4_1_len, dequant_q4_1_ದತ್ತಾಂಶ, "main", 2, 5 * ಗಾತ್ರ(uint32_t), {256 * 16, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_Q5_0], "dequant_q5_0", dequant_q5_0_len, dequant_q5_0_ದತ್ತಾಂಶ, "main", 2, 5 * ಗಾತ್ರ(uint32_t), {256 * 16, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_Q5_1], "dequant_q5_1", dequant_q5_1_len, dequant_q5_1_ದತ್ತಾಂಶ, "main", 2, 5 * ಗಾತ್ರ(uint32_t), {256 * 16, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_Q8_0], "dequant_q8_0", dequant_q8_0_len, dequant_q8_0_ದತ್ತಾಂಶ, "main", 2, 5 * ಗಾತ್ರ(uint32_t), {256 * 16, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant_transpose[GGML_TYPE_Q8_0], "dequant_q8_0_transpose", dequant_q8_0_transpose_len, dequant_q8_0_transpose_ದತ್ತಾಂಶ, "main", 2, 5 * ಗಾತ್ರ(uint32_t), {256 * 16, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_Q2_K], "dequant_q2_k", dequant_q2_k_len, dequant_q2_k_ದತ್ತಾಂಶ, "main", 2, 5 * ಗಾತ್ರ(uint32_t), {256 * 64, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_Q3_K], "dequant_q3_k", dequant_q3_k_len, dequant_q3_k_ದತ್ತಾಂಶ, "main", 2, 5 * ಗಾತ್ರ(uint32_t), {256 * 64, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_Q4_K], "dequant_q4_k", dequant_q4_k_len, dequant_q4_k_ದತ್ತಾಂಶ, "main", 2, 5 * ಗಾತ್ರ(uint32_t), {256 * 32, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_Q5_K], "dequant_q5_k", dequant_q5_k_len, dequant_q5_k_ದತ್ತಾಂಶ, "main", 2, 5 * ಗಾತ್ರ(uint32_t), {256 * 64, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_Q6_K], "dequant_q6_k", dequant_q6_k_len, dequant_q6_k_ದತ್ತಾಂಶ, "main", 2, 5 * ಗಾತ್ರ(uint32_t), {256 * 64, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_TQ1_0], "dequant_tq1_0", dequant_tq1_0_len, dequant_tq1_0_ದತ್ತಾಂಶ, "main", 2, 5 * ಗಾತ್ರ(uint32_t), {256 * 4, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_TQ2_0], "dequant_tq2_0", dequant_tq2_0_len, dequant_tq2_0_ದತ್ತಾಂಶ, "main", 2, 5 * ಗಾತ್ರ(uint32_t), {256 * 64, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_IQ1_S],   "dequant_iq1_s",   dequant_iq1_s_len,   dequant_iq1_s_ದತ್ತಾಂಶ,   "main", 2, 5 * ಗಾತ್ರ(uint32_t), {256 * 32, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_IQ1_M],   "dequant_iq1_m",   dequant_iq1_m_len,   dequant_iq1_m_ದತ್ತಾಂಶ,   "main", 2, 5 * ಗಾತ್ರ(uint32_t), {256 * 32, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_IQ2_XXS], "dequant_iq2_xxs", dequant_iq2_xxs_len, dequant_iq2_xxs_ದತ್ತಾಂಶ, "main", 2, 5 * ಗಾತ್ರ(uint32_t), {256 * 32, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_IQ2_XS],  "dequant_iq2_xs",  dequant_iq2_xs_len,  dequant_iq2_xs_ದತ್ತಾಂಶ,  "main", 2, 5 * ಗಾತ್ರ(uint32_t), {256 * 32, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_IQ2_S],   "dequant_iq2_s",   dequant_iq2_s_len,   dequant_iq2_s_ದತ್ತಾಂಶ,   "main", 2, 5 * ಗಾತ್ರ(uint32_t), {256 * 32, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_IQ3_XXS], "dequant_iq3_xxs", dequant_iq3_xxs_len, dequant_iq3_xxs_ದತ್ತಾಂಶ, "main", 2, 5 * ಗಾತ್ರ(uint32_t), {256 * 32, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_IQ3_S],   "dequant_iq3_s",   dequant_iq3_s_len,   dequant_iq3_s_ದತ್ತಾಂಶ,   "main", 2, 5 * ಗಾತ್ರ(uint32_t), {256 * 32, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_IQ4_XS],  "dequant_iq4_xs",  dequant_iq4_xs_len,  dequant_iq4_xs_ದತ್ತಾಂಶ,  "main", 2, 5 * ಗಾತ್ರ(uint32_t), {256 * 32, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_IQ4_NL],  "dequant_iq4_nl",  dequant_iq4_nl_len,  dequant_iq4_nl_ದತ್ತಾಂಶ,  "main", 2, 5 * ಗಾತ್ರ(uint32_t), {256 * 16, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_MXFP4],   "dequant_mxfp4",   dequant_mxfp4_len,   dequant_mxfp4_ದತ್ತಾಂಶ,   "main", 2, 5 * ಗಾತ್ರ(uint32_t), {256 * 16, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_NVFP4],   "dequant_nvfp4",   dequant_nvfp4_len,   dequant_nvfp4_ದತ್ತಾಂಶ,   "main", 2, 5 * ಗಾತ್ರ(uint32_t), {256 * 16, 1, 1}, {}, 1);

    // get_rows
    ggml_vk_create_pipeline(device, device->pipeline_get_ಅಡ್ಡಸಾಲುಗಳು[GGML_TYPE_F32 ], "get_rows_f32",  get_ಅಡ್ಡಸಾಲುಗಳು_f32_len,  get_ಅಡ್ಡಸಾಲುಗಳು_f32_ದತ್ತಾಂಶ,  "main", 3, ಗಾತ್ರ(vk_op_ದ್ವಿಮಾನ_ತಳ್ಳು_constants), { 512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_ಅಡ್ಡಸಾಲುಗಳು[GGML_TYPE_F16 ], "get_rows_f16",  get_ಅಡ್ಡಸಾಲುಗಳು_f16_len,  get_ಅಡ್ಡಸಾಲುಗಳು_f16_ದತ್ತಾಂಶ,  "main", 3, ಗಾತ್ರ(vk_op_ದ್ವಿಮಾನ_ತಳ್ಳು_constants), { 512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_ಅಡ್ಡಸಾಲುಗಳು[GGML_TYPE_BF16], "get_rows_bf16", get_ಅಡ್ಡಸಾಲುಗಳು_bf16_len, get_ಅಡ್ಡಸಾಲುಗಳು_bf16_ದತ್ತಾಂಶ, "main", 3, ಗಾತ್ರ(vk_op_ದ್ವಿಮಾನ_ತಳ್ಳು_constants), { 512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_ಅಡ್ಡಸಾಲುಗಳು[GGML_TYPE_Q1_0], "get_rows_q1_0", get_ಅಡ್ಡಸಾಲುಗಳು_q1_0_len, get_ಅಡ್ಡಸಾಲುಗಳು_q1_0_ದತ್ತಾಂಶ, "main", 3, ಗಾತ್ರ(vk_op_ದ್ವಿಮಾನ_ತಳ್ಳು_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_ಅಡ್ಡಸಾಲುಗಳು[GGML_TYPE_Q2_0], "get_rows_q2_0", get_ಅಡ್ಡಸಾಲುಗಳು_q2_0_len, get_ಅಡ್ಡಸಾಲುಗಳು_q2_0_ದತ್ತಾಂಶ, "main", 3, ಗಾತ್ರ(vk_op_ದ್ವಿಮಾನ_ತಳ್ಳು_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_ಅಡ್ಡಸಾಲುಗಳು[GGML_TYPE_Q4_0], "get_rows_q4_0", get_ಅಡ್ಡಸಾಲುಗಳು_q4_0_len, get_ಅಡ್ಡಸಾಲುಗಳು_q4_0_ದತ್ತಾಂಶ, "main", 3, ಗಾತ್ರ(vk_op_ದ್ವಿಮಾನ_ತಳ್ಳು_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_ಅಡ್ಡಸಾಲುಗಳು[GGML_TYPE_Q4_1], "get_rows_q4_1", get_ಅಡ್ಡಸಾಲುಗಳು_q4_1_len, get_ಅಡ್ಡಸಾಲುಗಳು_q4_1_ದತ್ತಾಂಶ, "main", 3, ಗಾತ್ರ(vk_op_ದ್ವಿಮಾನ_ತಳ್ಳು_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_ಅಡ್ಡಸಾಲುಗಳು[GGML_TYPE_Q5_0], "get_rows_q5_0", get_ಅಡ್ಡಸಾಲುಗಳು_q5_0_len, get_ಅಡ್ಡಸಾಲುಗಳು_q5_0_ದತ್ತಾಂಶ, "main", 3, ಗಾತ್ರ(vk_op_ದ್ವಿಮಾನ_ತಳ್ಳು_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_ಅಡ್ಡಸಾಲುಗಳು[GGML_TYPE_Q5_1], "get_rows_q5_1", get_ಅಡ್ಡಸಾಲುಗಳು_q5_1_len, get_ಅಡ್ಡಸಾಲುಗಳು_q5_1_ದತ್ತಾಂಶ, "main", 3, ಗಾತ್ರ(vk_op_ದ್ವಿಮಾನ_ತಳ್ಳು_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_ಅಡ್ಡಸಾಲುಗಳು[GGML_TYPE_Q8_0], "get_rows_q8_0", get_ಅಡ್ಡಸಾಲುಗಳು_q8_0_len, get_ಅಡ್ಡಸಾಲುಗಳು_q8_0_ದತ್ತಾಂಶ, "main", 3, ಗಾತ್ರ(vk_op_ದ್ವಿಮಾನ_ತಳ್ಳು_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_ಅಡ್ಡಸಾಲುಗಳು[GGML_TYPE_Q2_K], "get_rows_q2_k", get_ಅಡ್ಡಸಾಲುಗಳು_q2_k_len, get_ಅಡ್ಡಸಾಲುಗಳು_q2_k_ದತ್ತಾಂಶ, "main", 3, ಗಾತ್ರ(vk_op_ದ್ವಿಮಾನ_ತಳ್ಳು_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_ಅಡ್ಡಸಾಲುಗಳು[GGML_TYPE_Q3_K], "get_rows_q3_k", get_ಅಡ್ಡಸಾಲುಗಳು_q3_k_len, get_ಅಡ್ಡಸಾಲುಗಳು_q3_k_ದತ್ತಾಂಶ, "main", 3, ಗಾತ್ರ(vk_op_ದ್ವಿಮಾನ_ತಳ್ಳು_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_ಅಡ್ಡಸಾಲುಗಳು[GGML_TYPE_Q4_K], "get_rows_q4_k", get_ಅಡ್ಡಸಾಲುಗಳು_q4_k_len, get_ಅಡ್ಡಸಾಲುಗಳು_q4_k_ದತ್ತಾಂಶ, "main", 3, ಗಾತ್ರ(vk_op_ದ್ವಿಮಾನ_ತಳ್ಳು_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_ಅಡ್ಡಸಾಲುಗಳು[GGML_TYPE_Q5_K], "get_rows_q5_k", get_ಅಡ್ಡಸಾಲುಗಳು_q5_k_len, get_ಅಡ್ಡಸಾಲುಗಳು_q5_k_ದತ್ತಾಂಶ, "main", 3, ಗಾತ್ರ(vk_op_ದ್ವಿಮಾನ_ತಳ್ಳು_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_ಅಡ್ಡಸಾಲುಗಳು[GGML_TYPE_Q6_K], "get_rows_q6_k", get_ಅಡ್ಡಸಾಲುಗಳು_q6_k_len, get_ಅಡ್ಡಸಾಲುಗಳು_q6_k_ದತ್ತಾಂಶ, "main", 3, ಗಾತ್ರ(vk_op_ದ್ವಿಮಾನ_ತಳ್ಳು_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_ಅಡ್ಡಸಾಲುಗಳು[GGML_TYPE_TQ1_0], "get_rows_tq1_0", get_ಅಡ್ಡಸಾಲುಗಳು_tq1_0_len, get_ಅಡ್ಡಸಾಲುಗಳು_tq1_0_ದತ್ತಾಂಶ, "main", 3, ಗಾತ್ರ(vk_op_ದ್ವಿಮಾನ_ತಳ್ಳು_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_ಅಡ್ಡಸಾಲುಗಳು[GGML_TYPE_TQ2_0], "get_rows_tq2_0", get_ಅಡ್ಡಸಾಲುಗಳು_tq2_0_len, get_ಅಡ್ಡಸಾಲುಗಳು_tq2_0_ದತ್ತಾಂಶ, "main", 3, ಗಾತ್ರ(vk_op_ದ್ವಿಮಾನ_ತಳ್ಳು_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_ಅಡ್ಡಸಾಲುಗಳು[GGML_TYPE_IQ1_S],   "get_rows_iq1_s",   get_ಅಡ್ಡಸಾಲುಗಳು_iq1_s_len,   get_ಅಡ್ಡಸಾಲುಗಳು_iq1_s_ದತ್ತಾಂಶ,   "main", 3, ಗಾತ್ರ(vk_op_ದ್ವಿಮಾನ_ತಳ್ಳು_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_ಅಡ್ಡಸಾಲುಗಳು[GGML_TYPE_IQ1_M],   "get_rows_iq1_m",   get_ಅಡ್ಡಸಾಲುಗಳು_iq1_m_len,   get_ಅಡ್ಡಸಾಲುಗಳು_iq1_m_ದತ್ತಾಂಶ,   "main", 3, ಗಾತ್ರ(vk_op_ದ್ವಿಮಾನ_ತಳ್ಳು_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_ಅಡ್ಡಸಾಲುಗಳು[GGML_TYPE_IQ2_XXS], "get_rows_iq2_xxs", get_ಅಡ್ಡಸಾಲುಗಳು_iq2_xxs_len, get_ಅಡ್ಡಸಾಲುಗಳು_iq2_xxs_ದತ್ತಾಂಶ, "main", 3, ಗಾತ್ರ(vk_op_ದ್ವಿಮಾನ_ತಳ್ಳು_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_ಅಡ್ಡಸಾಲುಗಳು[GGML_TYPE_IQ2_XS],  "get_rows_iq2_xs",  get_ಅಡ್ಡಸಾಲುಗಳು_iq2_xs_len,  get_ಅಡ್ಡಸಾಲುಗಳು_iq2_xs_ದತ್ತಾಂಶ,  "main", 3, ಗಾತ್ರ(vk_op_ದ್ವಿಮಾನ_ತಳ್ಳು_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_ಅಡ್ಡಸಾಲುಗಳು[GGML_TYPE_IQ2_S],   "get_rows_iq2_s",   get_ಅಡ್ಡಸಾಲುಗಳು_iq2_s_len,   get_ಅಡ್ಡಸಾಲುಗಳು_iq2_s_ದತ್ತಾಂಶ,   "main", 3, ಗಾತ್ರ(vk_op_ದ್ವಿಮಾನ_ತಳ್ಳು_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_ಅಡ್ಡಸಾಲುಗಳು[GGML_TYPE_IQ3_XXS], "get_rows_iq3_xxs", get_ಅಡ್ಡಸಾಲುಗಳು_iq3_xxs_len, get_ಅಡ್ಡಸಾಲುಗಳು_iq3_xxs_ದತ್ತಾಂಶ, "main", 3, ಗಾತ್ರ(vk_op_ದ್ವಿಮಾನ_ತಳ್ಳು_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_ಅಡ್ಡಸಾಲುಗಳು[GGML_TYPE_IQ3_S],   "get_rows_iq3_s",   get_ಅಡ್ಡಸಾಲುಗಳು_iq3_s_len,   get_ಅಡ್ಡಸಾಲುಗಳು_iq3_s_ದತ್ತಾಂಶ,   "main", 3, ಗಾತ್ರ(vk_op_ದ್ವಿಮಾನ_ತಳ್ಳು_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_ಅಡ್ಡಸಾಲುಗಳು[GGML_TYPE_IQ4_XS],  "get_rows_iq4_xs",  get_ಅಡ್ಡಸಾಲುಗಳು_iq4_xs_len,  get_ಅಡ್ಡಸಾಲುಗಳು_iq4_xs_ದತ್ತಾಂಶ,  "main", 3, ಗಾತ್ರ(vk_op_ದ್ವಿಮಾನ_ತಳ್ಳು_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_ಅಡ್ಡಸಾಲುಗಳು[GGML_TYPE_IQ4_NL],  "get_rows_iq4_nl",  get_ಅಡ್ಡಸಾಲುಗಳು_iq4_nl_len,  get_ಅಡ್ಡಸಾಲುಗಳು_iq4_nl_ದತ್ತಾಂಶ,  "main", 3, ಗಾತ್ರ(vk_op_ದ್ವಿಮಾನ_ತಳ್ಳು_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_ಅಡ್ಡಸಾಲುಗಳು[GGML_TYPE_MXFP4],   "get_rows_mxfp4",   get_ಅಡ್ಡಸಾಲುಗಳು_mxfp4_len,   get_ಅಡ್ಡಸಾಲುಗಳು_mxfp4_ದತ್ತಾಂಶ,   "main", 3, ಗಾತ್ರ(vk_op_ದ್ವಿಮಾನ_ತಳ್ಳು_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_ಅಡ್ಡಸಾಲುಗಳು[GGML_TYPE_NVFP4],   "get_rows_nvfp4",   get_ಅಡ್ಡಸಾಲುಗಳು_nvfp4_len,   get_ಅಡ್ಡಸಾಲುಗಳು_nvfp4_ದತ್ತಾಂಶ,   "main", 3, ಗಾತ್ರ(vk_op_ದ್ವಿಮಾನ_ತಳ್ಳು_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_ಅಡ್ಡಸಾಲುಗಳು[GGML_TYPE_I32],     "get_rows_i32",     get_ಅಡ್ಡಸಾಲುಗಳು_i32_len,     get_ಅಡ್ಡಸಾಲುಗಳು_i32_ದತ್ತಾಂಶ,     "main", 3, ಗಾತ್ರ(vk_op_ದ್ವಿಮಾನ_ತಳ್ಳು_constants), {1024, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_get_ಅಡ್ಡಸಾಲುಗಳು_f32[GGML_TYPE_F32 ], "get_rows_f32_f32",  get_ಅಡ್ಡಸಾಲುಗಳು_f32_f32_len,  get_ಅಡ್ಡಸಾಲುಗಳು_f32_f32_ದತ್ತಾಂಶ,  "main", 3, ಗಾತ್ರ(vk_op_ದ್ವಿಮಾನ_ತಳ್ಳು_constants), { 512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_ಅಡ್ಡಸಾಲುಗಳು_f32[GGML_TYPE_F16 ], "get_rows_f16_f32",  get_ಅಡ್ಡಸಾಲುಗಳು_f16_f32_len,  get_ಅಡ್ಡಸಾಲುಗಳು_f16_f32_ದತ್ತಾಂಶ,  "main", 3, ಗಾತ್ರ(vk_op_ದ್ವಿಮಾನ_ತಳ್ಳು_constants), { 512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_ಅಡ್ಡಸಾಲುಗಳು_f32[GGML_TYPE_BF16], "get_rows_bf16_f32", get_ಅಡ್ಡಸಾಲುಗಳು_bf16_f32_len, get_ಅಡ್ಡಸಾಲುಗಳು_bf16_f32_ದತ್ತಾಂಶ, "main", 3, ಗಾತ್ರ(vk_op_ದ್ವಿಮಾನ_ತಳ್ಳು_constants), { 512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_ಅಡ್ಡಸಾಲುಗಳು_f32[GGML_TYPE_Q1_0], "get_rows_q1_0_f32", get_ಅಡ್ಡಸಾಲುಗಳು_q1_0_f32_len, get_ಅಡ್ಡಸಾಲುಗಳು_q1_0_f32_ದತ್ತಾಂಶ, "main", 3, ಗಾತ್ರ(vk_op_ದ್ವಿಮಾನ_ತಳ್ಳು_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_ಅಡ್ಡಸಾಲುಗಳು_f32[GGML_TYPE_Q2_0], "get_rows_q2_0_f32", get_ಅಡ್ಡಸಾಲುಗಳು_q2_0_f32_len, get_ಅಡ್ಡಸಾಲುಗಳು_q2_0_f32_ದತ್ತಾಂಶ, "main", 3, ಗಾತ್ರ(vk_op_ದ್ವಿಮಾನ_ತಳ್ಳು_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_ಅಡ್ಡಸಾಲುಗಳು_f32[GGML_TYPE_Q4_0], "get_rows_q4_0_f32", get_ಅಡ್ಡಸಾಲುಗಳು_q4_0_f32_len, get_ಅಡ್ಡಸಾಲುಗಳು_q4_0_f32_ದತ್ತಾಂಶ, "main", 3, ಗಾತ್ರ(vk_op_ದ್ವಿಮಾನ_ತಳ್ಳು_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_ಅಡ್ಡಸಾಲುಗಳು_f32[GGML_TYPE_Q4_1], "get_rows_q4_1_f32", get_ಅಡ್ಡಸಾಲುಗಳು_q4_1_f32_len, get_ಅಡ್ಡಸಾಲುಗಳು_q4_1_f32_ದತ್ತಾಂಶ, "main", 3, ಗಾತ್ರ(vk_op_ದ್ವಿಮಾನ_ತಳ್ಳು_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_ಅಡ್ಡಸಾಲುಗಳು_f32[GGML_TYPE_Q5_0], "get_rows_q5_0_f32", get_ಅಡ್ಡಸಾಲುಗಳು_q5_0_f32_len, get_ಅಡ್ಡಸಾಲುಗಳು_q5_0_f32_ದತ್ತಾಂಶ, "main", 3, ಗಾತ್ರ(vk_op_ದ್ವಿಮಾನ_ತಳ್ಳು_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_ಅಡ್ಡಸಾಲುಗಳು_f32[GGML_TYPE_Q5_1], "get_rows_q5_1_f32", get_ಅಡ್ಡಸಾಲುಗಳು_q5_1_f32_len, get_ಅಡ್ಡಸಾಲುಗಳು_q5_1_f32_ದತ್ತಾಂಶ, "main", 3, ಗಾತ್ರ(vk_op_ದ್ವಿಮಾನ_ತಳ್ಳು_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_ಅಡ್ಡಸಾಲುಗಳು_f32[GGML_TYPE_Q8_0], "get_rows_q8_0_f32", get_ಅಡ್ಡಸಾಲುಗಳು_q8_0_f32_len, get_ಅಡ್ಡಸಾಲುಗಳು_q8_0_f32_ದತ್ತಾಂಶ, "main", 3, ಗಾತ್ರ(vk_op_ದ್ವಿಮಾನ_ತಳ್ಳು_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_ಅಡ್ಡಸಾಲುಗಳು_f32[GGML_TYPE_Q2_K], "get_rows_q2_k_f32", get_ಅಡ್ಡಸಾಲುಗಳು_q2_k_f32_len, get_ಅಡ್ಡಸಾಲುಗಳು_q2_k_f32_ದತ್ತಾಂಶ, "main", 3, ಗಾತ್ರ(vk_op_ದ್ವಿಮಾನ_ತಳ್ಳು_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_ಅಡ್ಡಸಾಲುಗಳು_f32[GGML_TYPE_Q3_K], "get_rows_q3_k_f32", get_ಅಡ್ಡಸಾಲುಗಳು_q3_k_f32_len, get_ಅಡ್ಡಸಾಲುಗಳು_q3_k_f32_ದತ್ತಾಂಶ, "main", 3, ಗಾತ್ರ(vk_op_ದ್ವಿಮಾನ_ತಳ್ಳು_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_ಅಡ್ಡಸಾಲುಗಳು_f32[GGML_TYPE_Q4_K], "get_rows_q4_k_f32", get_ಅಡ್ಡಸಾಲುಗಳು_q4_k_f32_len, get_ಅಡ್ಡಸಾಲುಗಳು_q4_k_f32_ದತ್ತಾಂಶ, "main", 3, ಗಾತ್ರ(vk_op_ದ್ವಿಮಾನ_ತಳ್ಳು_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_ಅಡ್ಡಸಾಲುಗಳು_f32[GGML_TYPE_Q5_K], "get_rows_q5_k_f32", get_ಅಡ್ಡಸಾಲುಗಳು_q5_k_f32_len, get_ಅಡ್ಡಸಾಲುಗಳು_q5_k_f32_ದತ್ತಾಂಶ, "main", 3, ಗಾತ್ರ(vk_op_ದ್ವಿಮಾನ_ತಳ್ಳು_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_ಅಡ್ಡಸಾಲುಗಳು_f32[GGML_TYPE_Q6_K], "get_rows_q6_k_f32", get_ಅಡ್ಡಸಾಲುಗಳು_q6_k_f32_len, get_ಅಡ್ಡಸಾಲುಗಳು_q6_k_f32_ದತ್ತಾಂಶ, "main", 3, ಗಾತ್ರ(vk_op_ದ್ವಿಮಾನ_ತಳ್ಳು_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_ಅಡ್ಡಸಾಲುಗಳು_f32[GGML_TYPE_TQ1_0], "get_rows_tq1_0_f32", get_ಅಡ್ಡಸಾಲುಗಳು_tq1_0_f32_len, get_ಅಡ್ಡಸಾಲುಗಳು_tq1_0_f32_ದತ್ತಾಂಶ, "main", 3, ಗಾತ್ರ(vk_op_ದ್ವಿಮಾನ_ತಳ್ಳು_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_ಅಡ್ಡಸಾಲುಗಳು_f32[GGML_TYPE_TQ2_0], "get_rows_tq2_0_f32", get_ಅಡ್ಡಸಾಲುಗಳು_tq2_0_f32_len, get_ಅಡ್ಡಸಾಲುಗಳು_tq2_0_f32_ದತ್ತಾಂಶ, "main", 3, ಗಾತ್ರ(vk_op_ದ್ವಿಮಾನ_ತಳ್ಳು_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_ಅಡ್ಡಸಾಲುಗಳು_f32[GGML_TYPE_IQ1_S],   "get_rows_iq1_s_f32",   get_ಅಡ್ಡಸಾಲುಗಳು_iq1_s_f32_len,   get_ಅಡ್ಡಸಾಲುಗಳು_iq1_s_f32_ದತ್ತಾಂಶ,   "main", 3, ಗಾತ್ರ(vk_op_ದ್ವಿಮಾನ_ತಳ್ಳು_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_ಅಡ್ಡಸಾಲುಗಳು_f32[GGML_TYPE_IQ1_M],   "get_rows_iq1_m_f32",   get_ಅಡ್ಡಸಾಲುಗಳು_iq1_m_f32_len,   get_ಅಡ್ಡಸಾಲುಗಳು_iq1_m_f32_ದತ್ತಾಂಶ,   "main", 3, ಗಾತ್ರ(vk_op_ದ್ವಿಮಾನ_ತಳ್ಳು_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_ಅಡ್ಡಸಾಲುಗಳು_f32[GGML_TYPE_IQ2_XXS], "get_rows_iq2_xxs_f32", get_ಅಡ್ಡಸಾಲುಗಳು_iq2_xxs_f32_len, get_ಅಡ್ಡಸಾಲುಗಳು_iq2_xxs_f32_ದತ್ತಾಂಶ, "main", 3, ಗಾತ್ರ(vk_op_ದ್ವಿಮಾನ_ತಳ್ಳು_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_ಅಡ್ಡಸಾಲುಗಳು_f32[GGML_TYPE_IQ2_XS],  "get_rows_iq2_xs_f32",  get_ಅಡ್ಡಸಾಲುಗಳು_iq2_xs_f32_len,  get_ಅಡ್ಡಸಾಲುಗಳು_iq2_xs_f32_ದತ್ತಾಂಶ,  "main", 3, ಗಾತ್ರ(vk_op_ದ್ವಿಮಾನ_ತಳ್ಳು_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_ಅಡ್ಡಸಾಲುಗಳು_f32[GGML_TYPE_IQ2_S],   "get_rows_iq2_s_f32",   get_ಅಡ್ಡಸಾಲುಗಳು_iq2_s_f32_len,   get_ಅಡ್ಡಸಾಲುಗಳು_iq2_s_f32_ದತ್ತಾಂಶ,   "main", 3, ಗಾತ್ರ(vk_op_ದ್ವಿಮಾನ_ತಳ್ಳು_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_ಅಡ್ಡಸಾಲುಗಳು_f32[GGML_TYPE_IQ3_XXS], "get_rows_iq3_xxs_f32", get_ಅಡ್ಡಸಾಲುಗಳು_iq3_xxs_f32_len, get_ಅಡ್ಡಸಾಲುಗಳು_iq3_xxs_f32_ದತ್ತಾಂಶ, "main", 3, ಗಾತ್ರ(vk_op_ದ್ವಿಮಾನ_ತಳ್ಳು_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_ಅಡ್ಡಸಾಲುಗಳು_f32[GGML_TYPE_IQ3_S],   "get_rows_iq3_s_f32",   get_ಅಡ್ಡಸಾಲುಗಳು_iq3_s_f32_len,   get_ಅಡ್ಡಸಾಲುಗಳು_iq3_s_f32_ದತ್ತಾಂಶ,   "main", 3, ಗಾತ್ರ(vk_op_ದ್ವಿಮಾನ_ತಳ್ಳು_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_ಅಡ್ಡಸಾಲುಗಳು_f32[GGML_TYPE_IQ4_XS],  "get_rows_iq4_xs_f32",  get_ಅಡ್ಡಸಾಲುಗಳು_iq4_xs_f32_len,  get_ಅಡ್ಡಸಾಲುಗಳು_iq4_xs_f32_ದತ್ತಾಂಶ,  "main", 3, ಗಾತ್ರ(vk_op_ದ್ವಿಮಾನ_ತಳ್ಳು_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_ಅಡ್ಡಸಾಲುಗಳು_f32[GGML_TYPE_IQ4_NL],  "get_rows_iq4_nl_f32",  get_ಅಡ್ಡಸಾಲುಗಳು_iq4_nl_f32_len,  get_ಅಡ್ಡಸಾಲುಗಳು_iq4_nl_f32_ದತ್ತಾಂಶ,  "main", 3, ಗಾತ್ರ(vk_op_ದ್ವಿಮಾನ_ತಳ್ಳು_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_ಅಡ್ಡಸಾಲುಗಳು_f32[GGML_TYPE_MXFP4],   "get_rows_mxfp4_f32",   get_ಅಡ್ಡಸಾಲುಗಳು_mxfp4_f32_len,   get_ಅಡ್ಡಸಾಲುಗಳು_mxfp4_f32_ದತ್ತಾಂಶ,   "main", 3, ಗಾತ್ರ(vk_op_ದ್ವಿಮಾನ_ತಳ್ಳು_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_ಅಡ್ಡಸಾಲುಗಳು_f32[GGML_TYPE_NVFP4],   "get_rows_nvfp4_f32",   get_ಅಡ್ಡಸಾಲುಗಳು_nvfp4_f32_len,   get_ಅಡ್ಡಸಾಲುಗಳು_nvfp4_f32_ದತ್ತಾಂಶ,   "main", 3, ಗಾತ್ರ(vk_op_ದ್ವಿಮಾನ_ತಳ್ಳು_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_ಅಡ್ಡಸಾಲುಗಳು_ಹಿಂಭಾಗ_f32, "get_rows_back_f32", get_ಅಡ್ಡಸಾಲುಗಳು_ಹಿಂಭಾಗ_f32_len, get_ಅಡ್ಡಸಾಲುಗಳು_ಹಿಂಭಾಗ_f32_ದತ್ತಾಂಶ, "main", 3, ಗಾತ್ರ(vk_op_ದ್ವಿಮಾನ_ತಳ್ಳು_constants), {256, 1, 1}, {}, 1, ನಿಜ);

    ggml_vk_create_pipeline(device, device->pipeline_matmul_ವಿಭಜಿಸು_k_reduce, "split_k_reduce", ವಿಭಜಿಸು_k_reduce_len, ವಿಭಜಿಸು_k_reduce_ದತ್ತಾಂಶ, "main", 2, 2 * ಗಾತ್ರ(uint32_t), {256 * 4, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_flash_attn_ವಿಭಜಿಸು_k_reduce, "fa_split_k_reduce", fa_ವಿಭಜಿಸು_k_reduce_len, fa_ವಿಭಜಿಸು_k_reduce_ದತ್ತಾಂಶ, "main", 3, ಗಾತ್ರ(vk_op_flash_attn_ವಿಭಜಿಸು_k_reduce_ತಳ್ಳು_constants), {1, device->subgroup_ಗಾತ್ರ, 1}, {device->subgroup_ಗಾತ್ರ}, 1, ನಿಜ);

#ಆದರೆ defined(VK_KHR_cooperative_ಮಾತೃಕೆ) && defined(GGML_VULKAN_COOPMAT_GLSLC_SUPPORT)
    ಆದರೆ (device->vendor_id == VK_VENDOR_ID_INTEL && (device->architecture == INTEL_XE2 || (device->architecture == INTEL_XE1 && device->coopmat_support && device->uma))) {
        ಸ್ವಯಂಚಾಲಿತ ಊರ್ಧ್ವ_power_ದ_2 = [&](uint32_t ಒಳ) {
            GGML_ASSERT(ಒಳ != 0);
            ಆದರೆ (ಒಳ <= 1) ಹಿಂತಿರುಗಿ 1u;
            uint32_t ret = ಒಳ - 1;
            ret |= ret >> 1;
            ret |= ret >> 2;
            ret |= ret >> 4;
            ret |= ret >> 8;
            ret |= ret >> 16;
            ಹಿಂತಿರುಗಿ ret + 1;
            };

        uint32_t xe_native_sub_ಗುಂಪು_ಗಾತ್ರ = 16;
        ಆದರೆ (device->architecture == INTEL_XE1) {
            xe_native_sub_ಗುಂಪು_ಗಾತ್ರ = 8;
        }

        ಗಾಗಿ (ಸ್ವಯಂಚಾಲಿತ& it : device->pipeline_xe_fa_decode_dual_phases) {
            ಸ್ಥಿರಾಂಕ uint32_t ವಿಭಜಿಸು_p_chunk = 32;
            ಸ್ವಯಂಚಾಲಿತ HdQk = it.ಮೊದಲ;
            ಸ್ವಯಂಚಾಲಿತ& pipelines = it.second;
            uint32_t ತಲೆ_dim_qk = std::get<0>(HdQk);
            uint32_t ತಲೆ_dim_pv = std::get<1>(HdQk);
            uint32_t gqa_ಅನುಪಾತ = std::get<2>(HdQk);
            uint32_t q_len = std::get<3>(HdQk);
            ಸ್ಥಿರಾಂಕ uint32_t ಹೊರಹೋಗು_dim_ಪ್ರತಿ_wg = gqa_ಅನುಪಾತ > 16 ? 8 : 16;
            uint32_t aligned_q_len = ಊರ್ಧ್ವ_power_ದ_2(q_len);
            uint32_t ಗುಂಪು_sz_ph1 = std::min(std::max(aligned_q_len * xe_native_sub_ಗುಂಪು_ಗಾತ್ರ, 64u), 256u);
            uint32_t ಹೊರಹೋಗು_ಪ್ರತಿ_wg_ph1 = std::min(q_len, 256u / xe_native_sub_ಗುಂಪು_ಗಾತ್ರ);
            uint32_t aligned_gqa_ಅನುಪಾತ = ಊರ್ಧ್ವ_power_ದ_2(gqa_ಅನುಪಾತ);
            uint32_t ವಿಭಜಿಸು_p_ಪ್ರತಿ_iter_ph2 = 256;
            uint32_t ವಿಭಜಿಸು_p_ಪ್ರತಿ_warp = 16;
            uint32_t ಗುಂಪು_sz_ph2 = (ವಿಭಜಿಸು_p_ಪ್ರತಿ_iter_ph2 / ವಿಭಜಿಸು_p_ಪ್ರತಿ_warp) * xe_native_sub_ಗುಂಪು_ಗಾತ್ರ;
            uint32_t ಹೊರಹೋಗು_ಪ್ರತಿ_wg_ph2 = std::min(std::max(16u / aligned_gqa_ಅನುಪಾತ, 1u), q_len);
            ggml_vk_create_pipeline(device, pipelines.ಮೊದಲ, "xe_fa_decode_ph1", fa_decode_ph1_cm1_len, fa_decode_ph1_cm1_ದತ್ತಾಂಶ, "main", 5, ಗಾತ್ರ(vk_fa_xe_opt_ತಳ್ಳು_constants), { 1, 32, 1 }, { ಗುಂಪು_sz_ph1, gqa_ಅನುಪಾತ, ತಲೆ_dim_qk, xe_native_sub_ಗುಂಪು_ಗಾತ್ರ, ವಿಭಜಿಸು_p_chunk, ಹೊರಹೋಗು_ಪ್ರತಿ_wg_ph1 }, 1, ಸುಳ್ಳು, ನಿಜ, xe_native_sub_ಗುಂಪು_ಗಾತ್ರ);
            ggml_vk_create_pipeline(device, pipelines.second, "xe_fa_decode_ph2", fa_decode_ph2_cm1_len, fa_decode_ph2_cm1_ದತ್ತಾಂಶ, "main", 5, ಗಾತ್ರ(vk_fa_xe_opt_ತಳ್ಳು_constants), { 1, 1, 1 }, { ಗುಂಪು_sz_ph2, gqa_ಅನುಪಾತ, ತಲೆ_dim_pv, ಹೊರಹೋಗು_ಪ್ರತಿ_wg_ph2, xe_native_sub_ಗುಂಪು_ಗಾತ್ರ, ವಿಭಜಿಸು_p_ಪ್ರತಿ_iter_ph2, ವಿಭಜಿಸು_p_chunk, ಹೊರಹೋಗು_dim_ಪ್ರತಿ_wg }, 1, ಸುಳ್ಳು, ನಿಜ, xe_native_sub_ಗುಂಪು_ಗಾತ್ರ);
        }
    }
#ಆದರೆ_ಅಂತ್ಯ

    ಗಾಗಿ (ಸ್ವಯಂಚಾಲಿತ &it : device->pipeline_fa_ಮುಖವಾಡ_opt) {
        ಸ್ವಯಂಚಾಲಿತ BrBc = it.ಮೊದಲ;
        ggml_vk_create_pipeline(device, it.second, "fa_mask_opt", fa_ಮುಖವಾಡ_opt_len, fa_ಮುಖವಾಡ_opt_ದತ್ತಾಂಶ, "main", 2, ಗಾತ್ರ(vk_op_flash_attn_ಮುಖವಾಡ_opt_ತಳ್ಳು_constants), {1, 1, 1}, {128, 128 / device->subgroup_ಗಾತ್ರ, BrBc.ಮೊದಲ, BrBc.second}, 1, ನಿಜ, ನಿಜ, device->subgroup_ಗಾತ್ರ);
    }

    {
        // Large workgroup so the per-row KV scan parallelizes; capped to device limits.
        ಸ್ಥಿರಾಂಕ uint32_t compact_max = std::min({1024u, device->properties.limits.max_Compute_Work_ಗುಂಪು_Invocations, device->properties.limits.max_Compute_Work_ಗುಂಪು_ಗಾತ್ರ[0]});

        // Fast ballot prefix-sum path when the device supports full subgroups; otherwise
        // a shared-memory prefix-sum fallback. Both emit a deterministic ascending list.
        device->fa_sparse_compact_use_subgroups = device->subgroup_ballot && device->subgroup_require_ತುಂಬಿದ_support;
        ಆದರೆ (device->fa_sparse_compact_use_subgroups) {
            ಸ್ಥಿರಾಂಕ uint32_t compact_wg = std::max(device->subgroup_ಗಾತ್ರ, (compact_max / device->subgroup_ಗಾತ್ರ) * device->subgroup_ಗಾತ್ರ);
            ಸ್ಥಿರಾಂಕ uint32_t compact_ಅಂಕೆ_sg = compact_wg / device->subgroup_ಗಾತ್ರ;
            ggml_vk_create_pipeline(device, device->pipeline_fa_sparse_compact_subgroup, "fa_sparse_compact_subgroup", fa_sparse_compact_subgroup_len, fa_sparse_compact_subgroup_ದತ್ತಾಂಶ, "main", 2, ಗಾತ್ರ(vk_op_flash_attn_sparse_compact_ತಳ್ಳು_constants), {1, 1, 1}, {compact_wg, compact_ಅಂಕೆ_sg}, 1, ನಿಜ, ನಿಜ, device->subgroup_ಗಾತ್ರ);
        } ಇಲ್ಲದಿದ್ದರೆ {
            ggml_vk_create_pipeline(device, device->pipeline_fa_sparse_compact, "fa_sparse_compact", fa_sparse_compact_len, fa_sparse_compact_ದತ್ತಾಂಶ, "main", 2, ಗಾತ್ರ(vk_op_flash_attn_sparse_compact_ತಳ್ಳು_constants), {1, 1, 1}, {compact_max}, 1, ನಿಜ);
        }
    }

    ಆದರೆ (device->subgroup_clustered && device->subgroup_require_ತುಂಬಿದ_support) {
        ggml_vk_create_pipeline(device, device->pipeline_quantize_q8_1_x4, "quantize_q8_1_x4", quantize_q8_1_x4_subgroup_len, quantize_q8_1_x4_subgroup_ದತ್ತಾಂಶ, "main", 2, ಗಾತ್ರ(vk_quantize_q8_1_ತಳ್ಳು_constants), {32 * device->subgroup_ಗಾತ್ರ / 8, 1, 1}, { device->subgroup_ಗಾತ್ರ }, 1, ನಿಜ, ನಿಜ);
    } ಇಲ್ಲದಿದ್ದರೆ {
        ggml_vk_create_pipeline(device, device->pipeline_quantize_q8_1_x4, "quantize_q8_1_x4", quantize_q8_1_x4_len, quantize_q8_1_x4_ದತ್ತಾಂಶ, "main", 2, ಗಾತ್ರ(vk_quantize_q8_1_ತಳ್ಳು_constants), {32 * device->subgroup_ಗಾತ್ರ / 8, 1, 1}, { device->subgroup_ಗಾತ್ರ }, 1);
    }

    ಗಾಗಿ (uint32_t i = 0; i < p021_max_gqa_ಅನುಪಾತ; ++i) {
        ಆದರೆ (device->subgroup_arithmetic && device->subgroup_require_ತುಂಬಿದ_support) {
            ggml_vk_create_pipeline2(device, device->pipeline_mul_mat_vec_p021_f16_f32[i], "mul_mat_vec_p021_f16_f32"+std::to_string(i+1), mul_mat_vec_p021_f16_f32_subgroup_ಕೂಡಿಸು_len, mul_mat_vec_p021_f16_f32_subgroup_ಕೂಡಿಸು_ದತ್ತಾಂಶ, "main", mul_mat_vec_ಅಂಕೆ_bindings, ಗಾತ್ರ(vk_mat_vec_p021_ತಳ್ಳು_constants), {1, 1, 1}, {device->subgroup_ಗಾತ್ರ, i + 1}, 1, ನಿಜ, ನಿಜ);
        } ಇಲ್ಲದಿದ್ದರೆ {
            ggml_vk_create_pipeline2(device, device->pipeline_mul_mat_vec_p021_f16_f32[i], "mul_mat_vec_p021_f16_f32"+std::to_string(i+1), mul_mat_vec_p021_f16_f32_len,              mul_mat_vec_p021_f16_f32_ದತ್ತಾಂಶ,              "main", mul_mat_vec_ಅಂಕೆ_bindings, ಗಾತ್ರ(vk_mat_vec_p021_ತಳ್ಳು_constants), {1, 1, 1}, {device->subgroup_ಗಾತ್ರ, i + 1}, 1, ನಿಜ);
        }
    }
    ggml_vk_create_pipeline(device, device->pipeline_mul_mat_vec_nc_f16_f32, "mul_mat_vec_nc_f16_f32", mul_mat_vec_nc_f16_f32_len, mul_mat_vec_nc_f16_f32_ದತ್ತಾಂಶ, "main", mul_mat_vec_ಅಂಕೆ_bindings, ಗಾತ್ರ(vk_mat_vec_nc_ತಳ್ಳು_constants), {1, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_norm_f32, "norm_f32", norm_f32_len, norm_f32_ದತ್ತಾಂಶ, "main", 2, ಗಾತ್ರ(vk_op_unary_ತಳ್ಳು_constants), {1, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_ಗುಂಪು_norm_f32, "group_norm_f32", ಗುಂಪು_norm_f32_len, ಗುಂಪು_norm_f32_ದತ್ತಾಂಶ, "main", 2, ಗಾತ್ರ(vk_op_ತಳ್ಳು_constants), {1, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_rms_norm_f32, "rms_norm_f32", rms_norm_f32_len, rms_norm_f32_ದತ್ತಾಂಶ, "main", 4, ಗಾತ್ರ(vk_op_ದ್ವಿಮಾನ_ತಳ್ಳು_constants), {1, 1, 1}, {0, 0}, 1, ನಿಜ);
    ggml_vk_create_pipeline(device, device->pipeline_rms_norm_mul_f32, "rms_norm_mul_f32", rms_norm_f32_len, rms_norm_f32_ದತ್ತಾಂಶ, "main", 4, ಗಾತ್ರ(vk_op_ದ್ವಿಮಾನ_ತಳ್ಳು_constants), {1, 1, 1}, {0, 1}, 1, ನಿಜ);
    ggml_vk_create_pipeline(device, device->pipeline_rms_norm_mul_ಕೂಡಿಸು_f32, "rms_norm_mul_add_f32", rms_norm_mul_ಕೂಡಿಸು_f32_len, rms_norm_mul_ಕೂಡಿಸು_f32_ದತ್ತಾಂಶ, "main", 5, ಗಾತ್ರ(vk_op_ದ್ವಿಮಾನ_ತಳ್ಳು_constants), {1, 1, 1}, {0, 1, 0}, 1, ನಿಜ);
    ggml_vk_create_pipeline(device, device->pipeline_rms_norm_mul_ಕೂಡಿಸು_mul_f32, "rms_norm_mul_add_mul_f32", rms_norm_mul_ಕೂಡಿಸು_f32_len, rms_norm_mul_ಕೂಡಿಸು_f32_ದತ್ತಾಂಶ, "main", 5, ಗಾತ್ರ(vk_op_ದ್ವಿಮಾನ_ತಳ್ಳು_constants), {1, 1, 1}, {0, 1, 1}, 1, ನಿಜ);
    ggml_vk_create_pipeline(device, device->pipeline_rms_norm_mul_ಕೂಡಿಸು_partials_f32, "rms_norm_mul_add_partials_f32", rms_norm_mul_ಕೂಡಿಸು_partials_f32_len, rms_norm_mul_ಕೂಡಿಸು_partials_f32_ದತ್ತಾಂಶ, "main", 6, ಗಾತ್ರ(vk_op_ದ್ವಿಮಾನ_ತಳ್ಳು_constants), {1, 1, 1}, {0, 1, 0}, 1, ನಿಜ);
    ggml_vk_create_pipeline(device, device->pipeline_rms_norm_mul_ಕೂಡಿಸು_mul_partials_f32, "rms_norm_mul_add_mul_partials_f32", rms_norm_mul_ಕೂಡಿಸು_partials_f32_len, rms_norm_mul_ಕೂಡಿಸು_partials_f32_ದತ್ತಾಂಶ, "main", 6, ಗಾತ್ರ(vk_op_ದ್ವಿಮಾನ_ತಳ್ಳು_constants), {1, 1, 1}, {0, 1, 1}, 1, ನಿಜ);
    ggml_vk_create_pipeline(device, device->pipeline_rms_norm_ಗಣ_ಅಡ್ಡಸಾಲುಗಳು_f32_f32, "rms_norm_set_rows_f32_f32", rms_norm_ಗಣ_ಅಡ್ಡಸಾಲುಗಳು_f32_f32_len, rms_norm_ಗಣ_ಅಡ್ಡಸಾಲುಗಳು_f32_f32_ದತ್ತಾಂಶ, "main", 4, ಗಾತ್ರ(vk_op_ದ್ವಿಮಾನ_ತಳ್ಳು_constants), {1, 1, 1}, {0, 0}, 1, ನಿಜ);
    ggml_vk_create_pipeline(device, device->pipeline_rms_norm_ಗಣ_ಅಡ್ಡಸಾಲುಗಳು_f32_f16, "rms_norm_set_rows_f32_f16", rms_norm_ಗಣ_ಅಡ್ಡಸಾಲುಗಳು_f32_f16_len, rms_norm_ಗಣ_ಅಡ್ಡಸಾಲುಗಳು_f32_f16_ದತ್ತಾಂಶ, "main", 4, ಗಾತ್ರ(vk_op_ದ್ವಿಮಾನ_ತಳ್ಳು_constants), {1, 1, 1}, {0, 0}, 1, ನಿಜ);
    ggml_vk_create_pipeline(device, device->pipeline_rms_norm_partials_f32, "rms_norm_partials_f32", rms_norm_partials_f32_len, rms_norm_partials_f32_ದತ್ತಾಂಶ, "main", 4, ಗಾತ್ರ(vk_op_ದ್ವಿಮಾನ_ತಳ್ಳು_constants), {1, 1, 1}, {0, 0}, 1, ನಿಜ);
    ggml_vk_create_pipeline(device, device->pipeline_rms_norm_mul_partials_f32, "rms_norm_mul_partials_f32", rms_norm_partials_f32_len, rms_norm_partials_f32_ದತ್ತಾಂಶ, "main", 4, ಗಾತ್ರ(vk_op_ದ್ವಿಮಾನ_ತಳ್ಳು_constants), {1, 1, 1}, {0, 1}, 1, ನಿಜ);

    ಆದರೆ (ಗಾತ್ರ(vk_op_rms_norm_mul_rope_ತಳ್ಳು_constants) <= device->properties.limits.max_ತಳ್ಳು_Constants_ಗಾತ್ರ) {
        ggml_vk_create_pipeline(device, device->pipeline_rms_norm_mul_rope_f32_f32, "rms_norm_mul_rope_f32_f32", rms_norm_mul_rope_f32_f32_len, rms_norm_mul_rope_f32_f32_ದತ್ತಾಂಶ, "main", 7, ಗಾತ್ರ(vk_op_rms_norm_mul_rope_ತಳ್ಳು_constants), {1, 1, 1}, {0, 1}, 1, ನಿಜ);
        ggml_vk_create_pipeline(device, device->pipeline_rms_norm_mul_rope_f32_f16, "rms_norm_mul_rope_f32_f16", rms_norm_mul_rope_f32_f16_len, rms_norm_mul_rope_f32_f16_ದತ್ತಾಂಶ, "main", 7, ಗಾತ್ರ(vk_op_rms_norm_mul_rope_ತಳ್ಳು_constants), {1, 1, 1}, {0, 1}, 1, ನಿಜ);
    }

    ggml_vk_create_pipeline(device, device->pipeline_rms_norm_ಹಿಂಭಾಗ_f32, "rms_norm_back_f32", rms_norm_ಹಿಂಭಾಗ_f32_len, rms_norm_ಹಿಂಭಾಗ_f32_ದತ್ತಾಂಶ, "main", 3, ಗಾತ್ರ(vk_op_ತಳ್ಳು_constants), {1, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_l2_norm_f32, "l2_norm_f32", l2_norm_f32_len, l2_norm_f32_ದತ್ತಾಂಶ, "main", 2, ಗಾತ್ರ(vk_op_unary_ತಳ್ಳು_constants), {1, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_cpy_32, "cpy_32", cpy_32_len, cpy_32_ದತ್ತಾಂಶ, "main", 2, ಗಾತ್ರ(vk_op_unary_ತಳ್ಳು_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_f32_f16, "cpy_f32_f16", cpy_f32_f16_len, cpy_f32_f16_ದತ್ತಾಂಶ, "main", 2, ಗಾತ್ರ(vk_op_unary_ತಳ್ಳು_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_16, "cpy_16", cpy_16_len, cpy_16_ದತ್ತಾಂಶ, "main", 2, ಗಾತ್ರ(vk_op_unary_ತಳ್ಳು_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_f16_f32, "cpy_f16_f32", cpy_f16_f32_len, cpy_f16_f32_ದತ್ತಾಂಶ, "main", 2, ಗಾತ್ರ(vk_op_unary_ತಳ್ಳು_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_f32_bf16,"cpy_f32_bf16",cpy_f32_bf16_len,cpy_f32_bf16_ದತ್ತಾಂಶ,"main", 2, ಗಾತ್ರ(vk_op_unary_ತಳ್ಳು_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_bf16_f32,"cpy_bf16_f32",cpy_bf16_f32_len,cpy_bf16_f32_ದತ್ತಾಂಶ,"main", 2, ಗಾತ್ರ(vk_op_unary_ತಳ್ಳು_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_bf16_f16,"cpy_bf16_f16",cpy_bf16_f16_len,cpy_bf16_f16_ದತ್ತಾಂಶ,"main", 2, ಗಾತ್ರ(vk_op_unary_ತಳ್ಳು_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_i32_f32, "cpy_i32_f32", cpy_i32_f32_len, cpy_i32_f32_ದತ್ತಾಂಶ, "main", 2, ಗಾತ್ರ(vk_op_unary_ತಳ್ಳು_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_f32_i32, "cpy_f32_i32", cpy_f32_i32_len, cpy_f32_i32_ದತ್ತಾಂಶ, "main", 2, ಗಾತ್ರ(vk_op_unary_ತಳ್ಳು_constants), {512, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_contig_cpy_32, "contig_cpy_32", contig_cpy_32_len, contig_cpy_32_ದತ್ತಾಂಶ, "main", 2, ಗಾತ್ರ(vk_op_unary_ತಳ್ಳು_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_contig_cpy_f32_f16, "contig_cpy_f32_f16", contig_cpy_f32_f16_len, contig_cpy_f32_f16_ದತ್ತಾಂಶ, "main", 2, ಗಾತ್ರ(vk_op_unary_ತಳ್ಳು_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_contig_cpy_16, "contig_cpy_16", contig_cpy_16_len, contig_cpy_16_ದತ್ತಾಂಶ, "main", 2, ಗಾತ್ರ(vk_op_unary_ತಳ್ಳು_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_contig_cpy_f16_f32, "contig_cpy_f16_f32", contig_cpy_f16_f32_len, contig_cpy_f16_f32_ದತ್ತಾಂಶ, "main", 2, ಗಾತ್ರ(vk_op_unary_ತಳ್ಳು_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_contig_cpy_f32_bf16,"contig_cpy_f32_bf16",contig_cpy_f32_bf16_len,contig_cpy_f32_bf16_ದತ್ತಾಂಶ,"main", 2, ಗಾತ್ರ(vk_op_unary_ತಳ್ಳು_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_contig_cpy_bf16_f32,"contig_cpy_bf16_f32",contig_cpy_bf16_f32_len,contig_cpy_bf16_f32_ದತ್ತಾಂಶ,"main", 2, ಗಾತ್ರ(vk_op_unary_ತಳ್ಳು_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_contig_cpy_bf16_f16,"contig_cpy_bf16_f16",contig_cpy_bf16_f16_len,contig_cpy_bf16_f16_ದತ್ತಾಂಶ,"main", 2, ಗಾತ್ರ(vk_op_unary_ತಳ್ಳು_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_contig_cpy_i32_f32, "contig_cpy_i32_f32", contig_cpy_i32_f32_len, contig_cpy_i32_f32_ದತ್ತಾಂಶ, "main", 2, ಗಾತ್ರ(vk_op_unary_ತಳ್ಳು_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_contig_cpy_f32_i32, "contig_cpy_f32_i32", contig_cpy_f32_i32_len, contig_cpy_f32_i32_ದತ್ತಾಂಶ, "main", 2, ಗಾತ್ರ(vk_op_unary_ತಳ್ಳು_constants), {512, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_cpy_transpose_32, "cpy_transpose_32", cpy_transpose_32_len, cpy_transpose_32_ದತ್ತಾಂಶ, "main", 2, ಗಾತ್ರ(vk_op_unary_ತಳ್ಳು_constants), {1, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_transpose_16, "cpy_transpose_16", cpy_transpose_16_len, cpy_transpose_16_ದತ್ತಾಂಶ, "main", 2, ಗಾತ್ರ(vk_op_unary_ತಳ್ಳು_constants), {1, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_transpose_02_32, "cpy_transpose_02_32", cpy_transpose_02_32_len, cpy_transpose_02_32_ದತ್ತಾಂಶ, "main", 2, ಗಾತ್ರ(vk_op_unary_ತಳ್ಳು_constants), {1, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_transpose_02_16, "cpy_transpose_02_16", cpy_transpose_02_16_len, cpy_transpose_02_16_ದತ್ತಾಂಶ, "main", 2, ಗಾತ್ರ(vk_op_unary_ತಳ್ಳು_constants), {1, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_cpy_f32_quant[GGML_TYPE_Q1_0], "cpy_f32_q1_0", cpy_f32_q1_0_len, cpy_f32_q1_0_ದತ್ತಾಂಶ, "main", 2, ಗಾತ್ರ(vk_op_unary_ತಳ್ಳು_constants), {32, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_f32_quant[GGML_TYPE_Q2_0], "cpy_f32_q2_0", cpy_f32_q2_0_len, cpy_f32_q2_0_ದತ್ತಾಂಶ, "main", 2, ಗಾತ್ರ(vk_op_unary_ತಳ್ಳು_constants), {32, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_f32_quant[GGML_TYPE_Q4_0], "cpy_f32_q4_0", cpy_f32_q4_0_len, cpy_f32_q4_0_ದತ್ತಾಂಶ, "main", 2, ಗಾತ್ರ(vk_op_unary_ತಳ್ಳು_constants), {32, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_f32_quant[GGML_TYPE_Q4_1], "cpy_f32_q4_1", cpy_f32_q4_1_len, cpy_f32_q4_1_ದತ್ತಾಂಶ, "main", 2, ಗಾತ್ರ(vk_op_unary_ತಳ್ಳು_constants), {32, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_f32_quant[GGML_TYPE_Q5_0], "cpy_f32_q5_0", cpy_f32_q5_0_len, cpy_f32_q5_0_ದತ್ತಾಂಶ, "main", 2, ಗಾತ್ರ(vk_op_unary_ತಳ್ಳು_constants), {32, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_f32_quant[GGML_TYPE_Q5_1], "cpy_f32_q5_1", cpy_f32_q5_1_len, cpy_f32_q5_1_ದತ್ತಾಂಶ, "main", 2, ಗಾತ್ರ(vk_op_unary_ತಳ್ಳು_constants), {32, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_f32_quant[GGML_TYPE_Q8_0], "cpy_f32_q8_0", cpy_f32_q8_0_len, cpy_f32_q8_0_ದತ್ತಾಂಶ, "main", 2, ಗಾತ್ರ(vk_op_unary_ತಳ್ಳು_constants), {32, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_f32_quant[GGML_TYPE_IQ4_NL], "cpy_f32_iq4_nl", cpy_f32_iq4_nl_len, cpy_f32_iq4_nl_ದತ್ತಾಂಶ, "main", 2, ಗಾತ್ರ(vk_op_unary_ತಳ್ಳು_constants), {32, 1, 1}, {}, 1);

#ವ್ಯಾಖ್ಯಾನಿಸು ಗಣ_ಅಡ್ಡಸಾಲುಗಳು_2(src_idx, src, itype) \
        ggml_vk_create_pipeline(device, device->pipeline_ಗಣ_ಅಡ್ಡಸಾಲುಗಳು ## itype [src_idx][GGML_TYPE_F32],  "set_rows_" #src "_f32" #itype,  ಗಣ_ಅಡ್ಡಸಾಲುಗಳು_3 ## src ## _f32 ## itype ## _len,  ಗಣ_ಅಡ್ಡಸಾಲುಗಳು_3 ## src ## _f32 ## itype ## ದತ್ತಾಂಶ_2,  "main", 3, ಗಾತ್ರ(vk_op_ದ್ವಿಮಾನ_ತಳ್ಳು_constants), {1, 1, 1}, {1}, 1, ನಿಜ); \
        ggml_vk_create_pipeline(device, device->pipeline_ಗಣ_ಅಡ್ಡಸಾಲುಗಳು ## itype [src_idx][GGML_TYPE_F16],  "set_rows_" #src "_f16" #itype,  ಗಣ_ಅಡ್ಡಸಾಲುಗಳು_3 ## src ## _f16 ## itype ## _len,  ಗಣ_ಅಡ್ಡಸಾಲುಗಳು_3 ## src ## _f16 ## itype ## ದತ್ತಾಂಶ_2,  "main", 3, ಗಾತ್ರ(vk_op_ದ್ವಿಮಾನ_ತಳ್ಳು_constants), {1, 1, 1}, {1}, 1, ನಿಜ); \
        ggml_vk_create_pipeline(device, device->pipeline_ಗಣ_ಅಡ್ಡಸಾಲುಗಳು ## itype [src_idx][GGML_TYPE_BF16], "set_rows_" #src "_bf16" #itype, ಗಣ_ಅಡ್ಡಸಾಲುಗಳು_3 ## src ## _bf16 ## itype ## _len, ಗಣ_ಅಡ್ಡಸಾಲುಗಳು_3 ## src ## _bf16 ## itype ## ದತ್ತಾಂಶ_2, "main", 3, ಗಾತ್ರ(vk_op_ದ್ವಿಮಾನ_ತಳ್ಳು_constants), {1, 1, 1}, {1}, 1, ನಿಜ); \
        ggml_vk_create_pipeline(device, device->pipeline_ಗಣ_ಅಡ್ಡಸಾಲುಗಳು ## itype [src_idx][GGML_TYPE_Q1_0], "set_rows_" #src "_q1_0" #itype, ಗಣ_ಅಡ್ಡಸಾಲುಗಳು_3 ## src ## _q1_0 ## itype ## _len, ಗಣ_ಅಡ್ಡಸಾಲುಗಳು_3 ## src ## _q1_0 ## itype ## ದತ್ತಾಂಶ_2, "main", 3, ಗಾತ್ರ(vk_op_ದ್ವಿಮಾನ_ತಳ್ಳು_constants), {1, 1, 1}, {1}, 1, ನಿಜ); \
        ggml_vk_create_pipeline(device, device->pipeline_ಗಣ_ಅಡ್ಡಸಾಲುಗಳು ## itype [src_idx][GGML_TYPE_Q2_0], "set_rows_" #src "_q2_0" #itype, ಗಣ_ಅಡ್ಡಸಾಲುಗಳು_3 ## src ## _q2_0 ## itype ## _len, ಗಣ_ಅಡ್ಡಸಾಲುಗಳು_3 ## src ## _q2_0 ## itype ## ದತ್ತಾಂಶ_2, "main", 3, ಗಾತ್ರ(vk_op_ದ್ವಿಮಾನ_ತಳ್ಳು_constants), {1, 1, 1}, {1}, 1, ನಿಜ); \
        ggml_vk_create_pipeline(device, device->pipeline_ಗಣ_ಅಡ್ಡಸಾಲುಗಳು ## itype [src_idx][GGML_TYPE_Q4_0], "set_rows_" #src "_q4_0" #itype, ಗಣ_ಅಡ್ಡಸಾಲುಗಳು_3 ## src ## _q4_0 ## itype ## _len, ಗಣ_ಅಡ್ಡಸಾಲುಗಳು_3 ## src ## _q4_0 ## itype ## ದತ್ತಾಂಶ_2, "main", 3, ಗಾತ್ರ(vk_op_ದ್ವಿಮಾನ_ತಳ್ಳು_constants), {1, 1, 1}, {1}, 1, ನಿಜ); \
        ggml_vk_create_pipeline(device, device->pipeline_ಗಣ_ಅಡ್ಡಸಾಲುಗಳು ## itype [src_idx][GGML_TYPE_Q4_1], "set_rows_" #src "_q4_1" #itype, ಗಣ_ಅಡ್ಡಸಾಲುಗಳು_3 ## src ## _q4_1 ## itype ## _len, ಗಣ_ಅಡ್ಡಸಾಲುಗಳು_3 ## src ## _q4_1 ## itype ## ದತ್ತಾಂಶ_2, "main", 3, ಗಾತ್ರ(vk_op_ದ್ವಿಮಾನ_ತಳ್ಳು_constants), {1, 1, 1}, {1}, 1, ನಿಜ); \
        ggml_vk_create_pipeline(device, device->pipeline_ಗಣ_ಅಡ್ಡಸಾಲುಗಳು ## itype [src_idx][GGML_TYPE_Q5_0], "set_rows_" #src "_q5_0" #itype, ಗಣ_ಅಡ್ಡಸಾಲುಗಳು_3 ## src ## _q5_0 ## itype ## _len, ಗಣ_ಅಡ್ಡಸಾಲುಗಳು_3 ## src ## _q5_0 ## itype ## ದತ್ತಾಂಶ_2, "main", 3, ಗಾತ್ರ(vk_op_ದ್ವಿಮಾನ_ತಳ್ಳು_constants), {1, 1, 1}, {1}, 1, ನಿಜ); \
        ggml_vk_create_pipeline(device, device->pipeline_ಗಣ_ಅಡ್ಡಸಾಲುಗಳು ## itype [src_idx][GGML_TYPE_Q5_1], "set_rows_" #src "_q5_1" #itype, ಗಣ_ಅಡ್ಡಸಾಲುಗಳು_3 ## src ## _q5_1 ## itype ## _len, ಗಣ_ಅಡ್ಡಸಾಲುಗಳು_3 ## src ## _q5_1 ## itype ## ದತ್ತಾಂಶ_2, "main", 3, ಗಾತ್ರ(vk_op_ದ್ವಿಮಾನ_ತಳ್ಳು_constants), {1, 1, 1}, {1}, 1, ನಿಜ); \
        ggml_vk_create_pipeline(device, device->pipeline_ಗಣ_ಅಡ್ಡಸಾಲುಗಳು ## itype [src_idx][GGML_TYPE_Q8_0], "set_rows_" #src "_q8_0" #itype, ಗಣ_ಅಡ್ಡಸಾಲುಗಳು_3 ## src ## _q8_0 ## itype ## _len, ಗಣ_ಅಡ್ಡಸಾಲುಗಳು_3 ## src ## _q8_0 ## itype ## ದತ್ತಾಂಶ_2, "main", 3, ಗಾತ್ರ(vk_op_ದ್ವಿಮಾನ_ತಳ್ಳು_constants), {1, 1, 1}, {1}, 1, ನಿಜ); \
        ggml_vk_create_pipeline(device, device->pipeline_ಗಣ_ಅಡ್ಡಸಾಲುಗಳು ## itype [src_idx][GGML_TYPE_IQ4_NL], "set_rows_" #src "_iq4_nl" #itype, ಗಣ_ಅಡ್ಡಸಾಲುಗಳು_3 ## src ## _iq4_nl ## itype ## _len, ಗಣ_ಅಡ್ಡಸಾಲುಗಳು_3 ## src ## _iq4_nl ## itype ## ದತ್ತಾಂಶ_2, "main", 3, ಗಾತ್ರ(vk_op_ದ್ವಿಮಾನ_ತಳ್ಳು_constants), {1, 1, 1}, {1}, 1, ನಿಜ);

    ಗಣ_ಅಡ್ಡಸಾಲುಗಳು_2(0, f32, _i32)
    ಗಣ_ಅಡ್ಡಸಾಲುಗಳು_2(0, f32, _i64)
    ಗಣ_ಅಡ್ಡಸಾಲುಗಳು_2(1, f16, _i32)
    ಗಣ_ಅಡ್ಡಸಾಲುಗಳು_2(1, f16, _i64)
#ವ್ಯಾಖ್ಯಾನ_ರದ್ದು ಗಣ_ಅಡ್ಡಸಾಲುಗಳು_2


    ggml_vk_create_pipeline(device, device->pipeline_cpy_quant_f32[GGML_TYPE_Q1_0], "cpy_q1_0_f32", cpy_q1_0_f32_len, cpy_q1_0_f32_ದತ್ತಾಂಶ, "main", 2, ಗಾತ್ರ(vk_op_unary_ತಳ್ಳು_constants), {(uint32_t)ggml_blck_ಗಾತ್ರ(GGML_TYPE_Q1_0), 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_quant_f32[GGML_TYPE_Q2_0], "cpy_q2_0_f32", cpy_q2_0_f32_len, cpy_q2_0_f32_ದತ್ತಾಂಶ, "main", 2, ಗಾತ್ರ(vk_op_unary_ತಳ್ಳು_constants), {(uint32_t)ggml_blck_ಗಾತ್ರ(GGML_TYPE_Q2_0), 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_quant_f32[GGML_TYPE_Q4_0], "cpy_q4_0_f32", cpy_q4_0_f32_len, cpy_q4_0_f32_ದತ್ತಾಂಶ, "main", 2, ಗಾತ್ರ(vk_op_unary_ತಳ್ಳು_constants), {(uint32_t)ggml_blck_ಗಾತ್ರ(GGML_TYPE_Q4_0), 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_quant_f32[GGML_TYPE_Q4_1], "cpy_q4_1_f32", cpy_q4_1_f32_len, cpy_q4_1_f32_ದತ್ತಾಂಶ, "main", 2, ಗಾತ್ರ(vk_op_unary_ತಳ್ಳು_constants), {(uint32_t)ggml_blck_ಗಾತ್ರ(GGML_TYPE_Q4_1), 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_quant_f32[GGML_TYPE_Q5_0], "cpy_q5_0_f32", cpy_q5_0_f32_len, cpy_q5_0_f32_ದತ್ತಾಂಶ, "main", 2, ಗಾತ್ರ(vk_op_unary_ತಳ್ಳು_constants), {(uint32_t)ggml_blck_ಗಾತ್ರ(GGML_TYPE_Q5_0), 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_quant_f32[GGML_TYPE_Q5_1], "cpy_q5_1_f32", cpy_q5_1_f32_len, cpy_q5_1_f32_ದತ್ತಾಂಶ, "main", 2, ಗಾತ್ರ(vk_op_unary_ತಳ್ಳು_constants), {(uint32_t)ggml_blck_ಗಾತ್ರ(GGML_TYPE_Q5_1), 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_quant_f32[GGML_TYPE_Q8_0], "cpy_q8_0_f32", cpy_q8_0_f32_len, cpy_q8_0_f32_ದತ್ತಾಂಶ, "main", 2, ಗಾತ್ರ(vk_op_unary_ತಳ್ಳು_constants), {(uint32_t)ggml_blck_ಗಾತ್ರ(GGML_TYPE_Q8_0), 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_quant_f32[GGML_TYPE_IQ4_NL], "cpy_iq4_nl_f32", cpy_iq4_nl_f32_len, cpy_iq4_nl_f32_ದತ್ತಾಂಶ, "main", 2, ಗಾತ್ರ(vk_op_unary_ತಳ್ಳು_constants), {(uint32_t)ggml_blck_ಗಾತ್ರ(GGML_TYPE_IQ4_NL), 1, 1}, {}, 1);

    ಸ್ವಯಂಚಾಲಿತ get_suffix = [](ತಾರ್ಕಿಕ src0_f16, ತಾರ್ಕಿಕ src1_f16, ತಾರ್ಕಿಕ dst_f16) {
        std::string s;
        s += std::string(src0_f16 ? "_f16" : "_f32");
        s += std::string(src1_f16 ? "_f16" : "_f32");
        s += std::string(dst_f16 ? "_f16" : "_f32");
        ಹಿಂತಿರುಗಿ s;
    };

#ವ್ಯಾಖ್ಯಾನಿಸು CREATE_ದ್ವಿಮಾನ(name, namemod, spec, bindings) \
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ s0 : {0,1}) ಗಾಗಿ (ಪೂರ್ಣಾಂಕ s1 : {0,1}) ಗಾಗಿ (ಪೂರ್ಣಾಂಕ d : {0,1}) \
        ggml_vk_create_pipeline2(device, device->pipeline_ ## name ## namemod[s0][s1][d], \
                                #name + get_suffix(s0, s1, d) + #namemod, name ## _len[s0][s1][d], name ## ದತ್ತಾಂಶ_2[s0][s1][d], \
                                "main", (bindings), ಗಾತ್ರ(vk_op_ದ್ವಿಮಾನ_ತಳ್ಳು_constants), {512, 1, 1}, spec, 1);

    CREATE_ದ್ವಿಮಾನ(ಕೂಡಿಸು_2, , {0}, 4)
    CREATE_ದ್ವಿಮಾನ(ಕೂಡಿಸು_2, _norepeat, {1}, 4)
    CREATE_ದ್ವಿಮಾನ(sub, , {0}, 3)
    CREATE_ದ್ವಿಮಾನ(sub, _norepeat, {1}, 3)
    CREATE_ದ್ವಿಮಾನ(mul, , {0}, 3)
    CREATE_ದ್ವಿಮಾನ(mul, _norepeat, {1}, 3)
    CREATE_ದ್ವಿಮಾನ(div, , {0}, 3)
    CREATE_ದ್ವಿಮಾನ(div, _norepeat, {1}, 3)
    CREATE_ದ್ವಿಮಾನ(ಕೂಡಿಸು_rms, , {0}, 4)
    CREATE_ದ್ವಿಮಾನ(ಕೂಡಿಸು_rms, _norepeat, {1}, 4)
#ವ್ಯಾಖ್ಯಾನ_ರದ್ದು CREATE_ದ್ವಿಮಾನ

    ಆದರೆ (device->multi_ಕೂಡಿಸು) {
        ಗಾಗಿ (uint32_t i = 0; i < MAX_FUSED_ADDS; ++i) {
            ggml_vk_create_pipeline2(device, device->pipeline_multi_ಕೂಡಿಸು[i],     "multi_add_f32_"     + std::to_string(i+1), multi_ಕೂಡಿಸು_f32_len,     multi_ಕೂಡಿಸು_f32_ದತ್ತಾಂಶ,     "main", MAX_PARAMETER_ಎಣಿಕೆ, ಗಾತ್ರ(vk_op_multi_ಕೂಡಿಸು_ತಳ್ಳು_constants), {512, 1, 1}, {i+2}, 1);
            ggml_vk_create_pipeline2(device, device->pipeline_multi_ಕೂಡಿಸು_rms[i], "multi_add_rms_f32_" + std::to_string(i+1), multi_ಕೂಡಿಸು_rms_f32_len, multi_ಕೂಡಿಸು_rms_f32_ದತ್ತಾಂಶ, "main", MAX_PARAMETER_ಎಣಿಕೆ, ಗಾತ್ರ(vk_op_multi_ಕೂಡಿಸು_ತಳ್ಳು_constants), {512, 1, 1}, {i+2}, 1);
        }
    }

    ggml_vk_create_pipeline(device, device->pipeline_ಕೂಡಿಸು_id_f32, "add_id_f32", ಕೂಡಿಸು_id_f32_len, ಕೂಡಿಸು_id_f32_ದತ್ತಾಂಶ, "main", 4, ಗಾತ್ರ(vk_op_ಕೂಡಿಸು_id_ತಳ್ಳು_constants), {1, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_acc_f32, "acc_f32", acc_f32_len, acc_f32_ದತ್ತಾಂಶ, "main", 3, ಗಾತ್ರ(vk_op_ದ್ವಿಮಾನ_ತಳ್ಳು_constants), {512, 1, 1}, {0, 1}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_ಗಣ_f32, "set_f32", acc_f32_len, acc_f32_ದತ್ತಾಂಶ, "main", 3, ಗಾತ್ರ(vk_op_ದ್ವಿಮಾನ_ತಳ್ಳು_constants), {512, 1, 1}, {0, 0}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_concat_i8, "concat_i8", concat_i8_len, concat_i8_ದತ್ತಾಂಶ, "main", 3, ಗಾತ್ರ(vk_op_ದ್ವಿಮಾನ_ತಳ್ಳು_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_concat_i16, "concat_i16", concat_i16_len, concat_i16_ದತ್ತಾಂಶ, "main", 3, ಗಾತ್ರ(vk_op_ದ್ವಿಮಾನ_ತಳ್ಳು_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_concat_i32, "concat_i32", concat_i32_len, concat_i32_ದತ್ತಾಂಶ, "main", 3, ಗಾತ್ರ(vk_op_ದ್ವಿಮಾನ_ತಳ್ಳು_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_concat_i64, "concat_i64", concat_i64_len, concat_i64_ದತ್ತಾಂಶ, "main", 3, ಗಾತ್ರ(vk_op_ದ್ವಿಮಾನ_ತಳ್ಳು_constants), {512, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_upscale_ಹತ್ತಿರದ_f32, "upscale_f32", upscale_f32_len, upscale_f32_ದತ್ತಾಂಶ, "main", 2, ಗಾತ್ರ(vk_op_upscale_ತಳ್ಳು_constants), {512, 1, 1}, {GGML_SCALE_ಬಹುಲಕ_ಹತ್ತಿರದ}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_upscale_bilinear_f32, "upscale_f32", upscale_f32_len, upscale_f32_ದತ್ತಾಂಶ, "main", 2, ಗಾತ್ರ(vk_op_upscale_ತಳ್ಳು_constants), {512, 1, 1}, {GGML_SCALE_ಬಹುಲಕ_BILINEAR}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_upscale_bicubic_f32, "upscale_f32", upscale_f32_len, upscale_f32_ದತ್ತಾಂಶ, "main", 2, ಗಾತ್ರ(vk_op_upscale_ತಳ್ಳು_constants), {512, 1, 1}, {GGML_SCALE_ಬಹುಲಕ_BICUBIC}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_upscale_bilinear_antialias_f32, "upscale_f32", upscale_f32_len, upscale_f32_ದತ್ತಾಂಶ, "main", 2, ಗಾತ್ರ(vk_op_upscale_ತಳ್ಳು_constants), {512, 1, 1}, {GGML_SCALE_ಬಹುಲಕ_BILINEAR | GGML_SCALE_FLAG_ANTIALIAS}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_scale_f32, "scale_f32", scale_f32_len, scale_f32_ದತ್ತಾಂಶ, "main", 2, ಗಾತ್ರ(vk_op_unary_ತಳ್ಳು_constants), {512, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_log[0], "log_f32", log_f32_len, log_f32_ದತ್ತಾಂಶ, "main", 2, ಗಾತ್ರ(vk_op_unary_ತಳ್ಳು_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_log[1], "log_f16", log_f16_len, log_f16_ದತ್ತಾಂಶ, "main", 2, ಗಾತ್ರ(vk_op_unary_ತಳ್ಳು_constants), {512, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_tri[0], "tri_f32", tri_f32_len, tri_f32_ದತ್ತಾಂಶ, "main", 2, ಗಾತ್ರ(vk_op_unary_ತಳ್ಳು_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_tri[1], "tri_f16", tri_f16_len, tri_f16_ದತ್ತಾಂಶ, "main", 2, ಗಾತ್ರ(vk_op_unary_ತಳ್ಳು_constants), {512, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_diag[0], "diag_f32", diag_f32_len, diag_f32_ದತ್ತಾಂಶ, "main", 2, ಗಾತ್ರ(vk_op_unary_ತಳ್ಳು_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_diag[1], "diag_f16", diag_f16_len, diag_f16_ದತ್ತಾಂಶ, "main", 2, ಗಾತ್ರ(vk_op_unary_ತಳ್ಳು_constants), {512, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_pad_f32, "pad_f32", pad_f32_len, pad_f32_ದತ್ತಾಂಶ, "main", 2, ಗಾತ್ರ(vk_op_pad_ತಳ್ಳು_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_pad_reflect_1d_f32, "pad_reflect_1d_f32", pad_reflect_1d_f32_len, pad_reflect_1d_f32_ದತ್ತಾಂಶ, "main", 2, ಗಾತ್ರ(vk_op_unary_ತಳ್ಳು_constants), {512, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_roll_f32, "roll_f32", roll_f32_len, roll_f32_ದತ್ತಾಂಶ, "main", 2, ಗಾತ್ರ(vk_op_unary_ತಳ್ಳು_constants), {512, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_repeat_i32, "repeat_i32", repeat_i32_len, repeat_i32_ದತ್ತಾಂಶ, "main", 2, ಗಾತ್ರ(vk_op_unary_ತಳ್ಳು_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_repeat_ಹಿಂಭಾಗ_f32, "repeat_back_f32", repeat_ಹಿಂಭಾಗ_f32_len, repeat_ಹಿಂಭಾಗ_f32_ದತ್ತಾಂಶ, "main", 2, ಗಾತ್ರ(vk_op_unary_ತಳ್ಳು_constants), {512, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_repeat_i16, "repeat_i16", repeat_i16_len, repeat_i16_ದತ್ತಾಂಶ, "main", 2, ಗಾತ್ರ(vk_op_unary_ತಳ್ಳು_constants), {512, 1, 1}, {}, 1);

#ವ್ಯಾಖ್ಯಾನಿಸು CREATE_UNARY(name)  \
    ggml_vk_create_pipeline(device, device->pipeline_ ## name [0], #name "_f32", name ## _f32_len, name ## f32_ದತ್ತಾಂಶ, "main", 2, ಗಾತ್ರ(vk_op_unary_ತಳ್ಳು_constants), {512, 1, 1}, {}, 1);  \
    ggml_vk_create_pipeline(device, device->pipeline_ ## name [1], #name "_f16", name ## _f16_len, name ## f16_ದತ್ತಾಂಶ, "main", 2, ಗಾತ್ರ(vk_op_unary_ತಳ್ಳು_constants), {512, 1, 1}, {}, 1);

    CREATE_UNARY(elu)
    CREATE_UNARY(gelu)
    CREATE_UNARY(gelu_erf)
    CREATE_UNARY(gelu_ಶೀಘ್ರ)
    CREATE_UNARY(silu)
    CREATE_UNARY(relu)
    CREATE_UNARY(sqr)
    CREATE_UNARY(ವರ್ಗಮೂಲ)
    CREATE_UNARY(ಜ್ಯಾ)
    CREATE_UNARY(ಕೋಜ್ಯಾ)
    CREATE_UNARY(clamp)
    CREATE_UNARY(leaky_relu)
    CREATE_UNARY(xielu)
    CREATE_UNARY(neg)
    CREATE_UNARY(tanh)
    CREATE_UNARY(sigmoid)
    CREATE_UNARY(hardsigmoid)
    CREATE_UNARY(hardswish)
    CREATE_UNARY(ನಿರಪೇಕ್ಷ)
    CREATE_UNARY(softplus)
    CREATE_UNARY(ಹೆಜ್ಜೆ)
    CREATE_UNARY(round)
    CREATE_UNARY(ceil)
    CREATE_UNARY(floor)
    CREATE_UNARY(trunc)
    CREATE_UNARY(sgn)
    CREATE_UNARY(ಘಾತಾಂಕ)
    CREATE_UNARY(expm1)
#ವ್ಯಾಖ್ಯಾನ_ರದ್ದು CREATE_UNARY

// spec constants: {norepeat, op_on_b}
#ವ್ಯಾಖ್ಯಾನಿಸು CREATE_UNARY_MUL(name, idx) \
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ dt = 0; dt < 2; ++dt) { \
        ಸ್ಥಿರಾಂಕ size_t len_ = dt ? name ## _mul_f16_len : name ## _mul_f32_len; \
        ಸ್ಥಿರಾಂಕ ಚಿಹ್ನೆರಹಿತ ಅಕ್ಷರ * ದತ್ತಾಂಶ_3 = dt ? name ## mul_f16_ದತ್ತಾಂಶ : name ## mul_f32_ದತ್ತಾಂಶ; \
        ಸ್ಥಿರಾಂಕ std::string dts_ = dt ? "f16" : "f32"; \
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ ob = 0; ob < 2; ++ob) \
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ nr = 0; nr < 2; ++nr) \
                ggml_vk_create_pipeline(device, device->pipeline_unary_mul[(idx)][dt][nr][ob], \
                    (#name "_mul" + std::string(ob ? "_b" : "") + "_" + dts_ + (nr ? "_norepeat" : "")).c_str(), \
                    len_, ದತ್ತಾಂಶ_3, "main", 3, ಗಾತ್ರ(vk_op_ದ್ವಿಮಾನ_ತಳ್ಳು_constants), {512, 1, 1}, \
                    { (uint32_t) nr, (uint32_t) ob }, 1); \
    }

    CREATE_UNARY_MUL(gelu, 0)
    CREATE_UNARY_MUL(sigmoid, 1)
    CREATE_UNARY_MUL(silu, 2)
    CREATE_UNARY_MUL(softplus, 3)
#ವ್ಯಾಖ್ಯಾನ_ರದ್ದು CREATE_UNARY_MUL

    ggml_vk_create_pipeline(device, device->pipeline_add1_f16_f16, "add1_f16_f16", add1_f16_f16_len, add1_f16_f16_ದತ್ತಾಂಶ, "main", 3, ಗಾತ್ರ(vk_op_ದ್ವಿಮಾನ_ತಳ್ಳು_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_add1_f16_f32, "add1_f16_f32", add1_f16_f32_len, add1_f16_f32_ದತ್ತಾಂಶ, "main", 3, ಗಾತ್ರ(vk_op_ದ್ವಿಮಾನ_ತಳ್ಳು_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_add1_f32_f32, "add1_f32_f32", add1_f32_f32_len, add1_f32_f32_ದತ್ತಾಂಶ, "main", 3, ಗಾತ್ರ(vk_op_ದ್ವಿಮಾನ_ತಳ್ಳು_constants), {512, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_arange_f32, "arange_f32", arange_f32_len, arange_f32_ದತ್ತಾಂಶ, "main", 1, ಗಾತ್ರ(vk_op_ತಳ್ಳು_constants), {512, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_fill_f32, "fill_f32", fill_f32_len, fill_f32_ದತ್ತಾಂಶ, "main", 1, ಗಾತ್ರ(vk_op_ತಳ್ಳು_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_fill_f16, "fill_f16", fill_f16_len, fill_f16_ದತ್ತಾಂಶ, "main", 1, ಗಾತ್ರ(vk_op_ತಳ್ಳು_constants), {512, 1, 1}, {}, 1);

#ವ್ಯಾಖ್ಯಾನಿಸು CREATE_GLU(name)  \
    ggml_vk_create_pipeline(device, device->pipeline_ ## name [0], #name "_f32", name ## _f32_len, name ## f32_ದತ್ತಾಂಶ, "main", 3, ಗಾತ್ರ(vk_op_glu_ತಳ್ಳು_constants), {512, 1, 1}, {}, 1, ನಿಜ);   \
    ggml_vk_create_pipeline(device, device->pipeline_ ## name [1], #name "_f16", name ## _f16_len, name ## f16_ದತ್ತಾಂಶ, "main", 3, ಗಾತ್ರ(vk_op_glu_ತಳ್ಳು_constants), {512, 1, 1}, {}, 1, ನಿಜ);

    CREATE_GLU(geglu)
    CREATE_GLU(reglu)
    CREATE_GLU(swiglu)
    CREATE_GLU(swiglu_oai)
    CREATE_GLU(swiglu_clamp)
    CREATE_GLU(geglu_erf)
    CREATE_GLU(geglu_ಶೀಘ್ರ)
#ವ್ಯಾಖ್ಯಾನ_ರದ್ದು CREATE_GLU

    ggml_vk_create_pipeline(device, device->pipeline_silu_ಹಿಂಭಾಗ_f32, "silu_back_f32", silu_ಹಿಂಭಾಗ_f32_len, silu_ಹಿಂಭಾಗ_f32_ದತ್ತಾಂಶ, "main", 3, ಗಾತ್ರ(vk_op_ತಳ್ಳು_constants), {512, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_diag_ಮುಖವಾಡ_inf_f32, "diag_mask_inf_f32", diag_ಮುಖವಾಡ_inf_f32_len, diag_ಮುಖವಾಡ_inf_f32_ದತ್ತಾಂಶ, "main", 2, ಗಾತ್ರ(vk_op_diag_ಮುಖವಾಡ_ತಳ್ಳು_constants), {1, 512, 1}, {}, 1, ನಿಜ);

    ggml_vk_create_pipeline(device, device->pipeline_soft_max_f32, "soft_max_f32", soft_max_f32_len, soft_max_f32_ದತ್ತಾಂಶ, "main", 4, ಗಾತ್ರ(vk_op_soft_max_ತಳ್ಳು_constants), {1, 1, 1}, { device->subgroup_ಗಾತ್ರ }, 1);
    ggml_vk_create_pipeline(device, device->pipeline_soft_max_f32_wg512, "soft_max_f32_wg512", soft_max_f32_len, soft_max_f32_ದತ್ತಾಂಶ, "main", 4, ಗಾತ್ರ(vk_op_soft_max_ತಳ್ಳು_constants), {1, 1, 1}, { 512 }, 1);
    ggml_vk_create_pipeline(device, device->pipeline_soft_max_f32_f16, "soft_max_f32_f16", soft_max_f32_f16_len, soft_max_f32_f16_ದತ್ತಾಂಶ, "main", 4, ಗಾತ್ರ(vk_op_soft_max_ತಳ್ಳು_constants), {1, 1, 1}, { device->subgroup_ಗಾತ್ರ }, 1);
    ggml_vk_create_pipeline(device, device->pipeline_soft_max_f32_f16_wg512, "soft_max_f32_f16_wg512", soft_max_f32_f16_len, soft_max_f32_f16_ದತ್ತಾಂಶ, "main", 4, ಗಾತ್ರ(vk_op_soft_max_ತಳ್ಳು_constants), {1, 1, 1}, { 512 }, 1);
    ggml_vk_create_pipeline(device, device->pipeline_soft_max_ಹಿಂಭಾಗ_f32, "soft_max_back_f32", soft_max_ಹಿಂಭಾಗ_f32_len, soft_max_ಹಿಂಭಾಗ_f32_ದತ್ತಾಂಶ, "main", 3, ಗಾತ್ರ(vk_op_ತಳ್ಳು_constants), {1, 1, 1}, { device->subgroup_ಗಾತ್ರ }, 1, ನಿಜ);

    ggml_vk_create_pipeline(device, device->pipeline_soft_max_large1_f32,     "soft_max_large1_f32",     soft_max_large1_f32_len,     soft_max_large1_f32_ದತ್ತಾಂಶ,     "main", 6, ಗಾತ್ರ(vk_op_soft_max_ತಳ್ಳು_constants), {1, 1, 1}, { 128, 4 }, 1, ನಿಜ);
    ggml_vk_create_pipeline(device, device->pipeline_soft_max_large2_f32,     "soft_max_large2_f32",     soft_max_large2_f32_len,     soft_max_large2_f32_ದತ್ತಾಂಶ,     "main", 6, ಗಾತ್ರ(vk_op_soft_max_ತಳ್ಳು_constants), {1, 1, 1}, { 128, 4 }, 1, ನಿಜ);
    ggml_vk_create_pipeline(device, device->pipeline_soft_max_large3_f32,     "soft_max_large3_f32",     soft_max_large3_f32_len,     soft_max_large3_f32_ದತ್ತಾಂಶ,     "main", 6, ಗಾತ್ರ(vk_op_soft_max_ತಳ್ಳು_constants), {1, 1, 1}, { 128, 4 }, 1, ನಿಜ);
    ggml_vk_create_pipeline(device, device->pipeline_soft_max_large1_f32_f16, "soft_max_large1_f32_f16", soft_max_large1_f32_f16_len, soft_max_large1_f32_f16_ದತ್ತಾಂಶ, "main", 6, ಗಾತ್ರ(vk_op_soft_max_ತಳ್ಳು_constants), {1, 1, 1}, { 128, 4 }, 1, ನಿಜ);
    ggml_vk_create_pipeline(device, device->pipeline_soft_max_large2_f32_f16, "soft_max_large2_f32_f16", soft_max_large2_f32_f16_len, soft_max_large2_f32_f16_ದತ್ತಾಂಶ, "main", 6, ಗಾತ್ರ(vk_op_soft_max_ತಳ್ಳು_constants), {1, 1, 1}, { 128, 4 }, 1, ನಿಜ);
    ggml_vk_create_pipeline(device, device->pipeline_soft_max_large3_f32_f16, "soft_max_large3_f32_f16", soft_max_large3_f32_f16_len, soft_max_large3_f32_f16_ದತ್ತಾಂಶ, "main", 6, ಗಾತ್ರ(vk_op_soft_max_ತಳ್ಳು_constants), {1, 1, 1}, { 128, 4 }, 1, ನಿಜ);

    ggml_vk_create_pipeline(device, device->pipeline_rope_norm_f32, "rope_norm_f32", rope_norm_f32_len, rope_norm_f32_ದತ್ತಾಂಶ, "main", 5, ಗಾತ್ರ(vk_op_rope_ತಳ್ಳು_constants), {1, 512, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_rope_neox_f32, "rope_neox_f32", rope_neox_f32_len, rope_neox_f32_ದತ್ತಾಂಶ, "main", 5, ಗಾತ್ರ(vk_op_rope_ತಳ್ಳು_constants), {1, 512, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_rope_multi_f32, "rope_multi_f32", rope_multi_f32_len, rope_multi_f32_ದತ್ತಾಂಶ, "main", 5, ಗಾತ್ರ(vk_op_rope_ತಳ್ಳು_constants), {1, 512, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_rope_vision_f32, "rope_vision_f32", rope_vision_f32_len, rope_vision_f32_ದತ್ತಾಂಶ, "main", 5, ಗಾತ್ರ(vk_op_rope_ತಳ್ಳು_constants), {1, 512, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_rope_norm_f16, "rope_norm_f16", rope_norm_f16_len, rope_norm_f16_ದತ್ತಾಂಶ, "main", 5, ಗಾತ್ರ(vk_op_rope_ತಳ್ಳು_constants), {1, 512, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_rope_neox_f16, "rope_neox_f16", rope_neox_f16_len, rope_neox_f16_ದತ್ತಾಂಶ, "main", 5, ಗಾತ್ರ(vk_op_rope_ತಳ್ಳು_constants), {1, 512, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_rope_multi_f16, "rope_multi_f16", rope_multi_f16_len, rope_multi_f16_ದತ್ತಾಂಶ, "main", 5, ಗಾತ್ರ(vk_op_rope_ತಳ್ಳು_constants), {1, 512, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_rope_vision_f16, "rope_vision_f16", rope_vision_f16_len, rope_vision_f16_ದತ್ತಾಂಶ, "main", 5, ಗಾತ್ರ(vk_op_rope_ತಳ್ಳು_constants), {1, 512, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_rope_norm_f32_f16, "rope_norm_f32_f16", rope_norm_f32_f16_len, rope_norm_f32_f16_ದತ್ತಾಂಶ, "main", 5, ಗಾತ್ರ(vk_op_rope_ತಳ್ಳು_constants), {1, 512, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_rope_neox_f32_f16, "rope_neox_f32_f16", rope_neox_f32_f16_len, rope_neox_f32_f16_ದತ್ತಾಂಶ, "main", 5, ಗಾತ್ರ(vk_op_rope_ತಳ್ಳು_constants), {1, 512, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_rope_multi_f32_f16, "rope_multi_f32_f16", rope_multi_f32_f16_len, rope_multi_f32_f16_ದತ್ತಾಂಶ, "main", 5, ಗಾತ್ರ(vk_op_rope_ತಳ್ಳು_constants), {1, 512, 1}, {}, 1);

    ಗಾಗಿ (uint32_t i = 0; i < ಅಂಕೆ_argsort_pipelines; ++i) {
        uint32_t ಖಂಡ_ಗಾತ್ರ_2 = 1u << std::min(i, device->max_workgroup_ಗಾತ್ರ_log2);
        ಆದರೆ (i <= device->max_workgroup_ಗಾತ್ರ_log2 &&
            2 * ಗಾತ್ರ(ಪೂರ್ಣಾಂಕ) * ಖಂಡ_ಗಾತ್ರ_2 <= device->properties.limits.max_Compute_Shared_ಸ್ಮರಣೆ_ಗಾತ್ರ) {
            ಸ್ಥಿರಾಂಕ uint32_t NCOLS_PADDED_LOG2 = i;
            ggml_vk_create_pipeline2(device, device->pipeline_argsort_f32[i], "argsort_f32_"+std::to_string(i), argsort_f32_len, argsort_f32_ದತ್ತಾಂಶ, "main", 3, ಗಾತ್ರ(vk_op_argsort_ತಳ್ಳು_constants), {ಖಂಡ_ಗಾತ್ರ_2, 1, 1}, {ಖಂಡ_ಗಾತ್ರ_2, NCOLS_PADDED_LOG2}, 1, ನಿಜ);
        }
        ಸ್ಥಿರಾಂಕ uint32_t WG_UNROLL_ಗುಣಕ = ಖಂಡ_ಗಾತ್ರ_2 > 1 ? 2 : 1;
        ಖಂಡ_ಗಾತ್ರ_2 /= WG_UNROLL_ಗುಣಕ;
        ggml_vk_create_pipeline2(device, device->pipeline_argsort_large_f32[i], "argsort_large_f32_"+std::to_string(i), argsort_large_f32_len, argsort_large_f32_ದತ್ತಾಂಶ, "main", 3, ಗಾತ್ರ(vk_op_argsort_ತಳ್ಳು_constants), {ಖಂಡ_ಗಾತ್ರ_2 * WG_UNROLL_ಗುಣಕ, 1, 1}, {ಖಂಡ_ಗಾತ್ರ_2, WG_UNROLL_ಗುಣಕ}, 1, ನಿಜ);
    }

    ಗಾಗಿ (uint32_t i = 0; i < ಅಂಕೆ_topk_pipelines; ++i) {
        ಸ್ಥಿರಾಂಕ uint32_t ಖಂಡ_ಗಾತ್ರ_2 = 1u << i;
        ಸ್ಥಿರಾಂಕ uint32_t NCOLS_PADDED_LOG2 = i;
        ಆದರೆ (i <= device->max_workgroup_ಗಾತ್ರ_log2) {
            uint32_t nary_shmem = 2 * ಗಾತ್ರ(ಪೂರ್ಣಾಂಕ) * ಖಂಡ_ಗಾತ್ರ_2 +
                                  ಗಾತ್ರ(ಪೂರ್ಣಾಂಕ) * device->subgroup_ಗಾತ್ರ +
                                  2 * ಗಾತ್ರ(ಪೂರ್ಣಾಂಕ) +
                                  2 * (ಖಂಡ_ಗಾತ್ರ_2 / device->subgroup_ಗಾತ್ರ) * ಗಾತ್ರ(ಪೂರ್ಣಾಂಕ);
            ಆದರೆ (device->subgroup_arithmetic && device->subgroup_require_ತುಂಬಿದ_support && device->subgroup_shuffle && device->subgroup_ballot &&
                nary_shmem <= device->properties.limits.max_Compute_Shared_ಸ್ಮರಣೆ_ಗಾತ್ರ) {
                ggml_vk_create_pipeline2(device, device->pipeline_topk_f32[i], "topk_f32_"+std::to_string(i), topk_nary_ಹುಡುಕು_f32_len, topk_nary_ಹುಡುಕು_f32_ದತ್ತಾಂಶ, "main", 2, ಗಾತ್ರ(vk_op_topk_ತಳ್ಳು_constants), {ಖಂಡ_ಗಾತ್ರ_2, 1, 1}, {ಖಂಡ_ಗಾತ್ರ_2, device->subgroup_ಗಾತ್ರ, device->subgroup_ಗಾತ್ರ_log2}, 1, ನಿಜ, ನಿಜ, device->subgroup_ಗಾತ್ರ);
            } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (2 * ಗಾತ್ರ(ಪೂರ್ಣಾಂಕ) * ಖಂಡ_ಗಾತ್ರ_2 <= device->properties.limits.max_Compute_Shared_ಸ್ಮರಣೆ_ಗಾತ್ರ) {
                ggml_vk_create_pipeline2(device, device->pipeline_topk_f32[i], "topk_f32_"+std::to_string(i), topk_argsort_f32_len, topk_argsort_f32_ದತ್ತಾಂಶ, "main", 2, ಗಾತ್ರ(vk_op_topk_ತಳ್ಳು_constants), {ಖಂಡ_ಗಾತ್ರ_2, 1, 1}, {ಖಂಡ_ಗಾತ್ರ_2, NCOLS_PADDED_LOG2}, 1, ನಿಜ);
            }
        }
    }

    // large-k fallback: one workgroup per row, radix-select instead of a full sort. The QSA
    // variant (spec constant 1) additionally gathers the qwen4 indexer input on the fly.
    {
        ಸ್ಥಿರಾಂಕ uint32_t ಖಂಡ_ಗಾತ್ರ_2 = 1u << std::min(10u, device->max_workgroup_ಗಾತ್ರ_log2);
        ggml_vk_create_pipeline2(device, device->pipeline_topk_radix_f32, "topk_radix_f32", topk_radix_select_f32_len, topk_radix_select_f32_ದತ್ತಾಂಶ, "main", 5, ಗಾತ್ರ(vk_op_topk_radix_ತಳ್ಳು_constants), {ಖಂಡ_ಗಾತ್ರ_2, 1, 1}, {ಖಂಡ_ಗಾತ್ರ_2, 0}, 1, ನಿಜ);
        ggml_vk_create_pipeline2(device, device->pipeline_topk_radix_qsa, "topk_radix_qsa", topk_radix_select_f32_len, topk_radix_select_f32_ದತ್ತಾಂಶ, "main", 5, ಗಾತ್ರ(vk_op_topk_radix_ತಳ್ಳು_constants), {ಖಂಡ_ಗಾತ್ರ_2, 1, 1}, {ಖಂಡ_ಗಾತ್ರ_2, 1}, 1, ನಿಜ);
    }

    ggml_vk_create_pipeline(device, device->pipeline_argmax_f32, "argmax_f32", argmax_f32_len, argmax_f32_ದತ್ತಾಂಶ, "main", 2, ಗಾತ್ರ(vk_op_ತಳ್ಳು_constants), {1, 1, 1}, { device->subgroup_ಗಾತ್ರ }, 1);

    ggml_vk_create_pipeline(device, device->pipeline_ಮೊತ್ತ_ಅಡ್ಡಸಾಲುಗಳು_f32, "sum_rows_f32", ಮೊತ್ತ_ಅಡ್ಡಸಾಲುಗಳು_f32_len, ಮೊತ್ತ_ಅಡ್ಡಸಾಲುಗಳು_f32_ದತ್ತಾಂಶ, "main", 2, ಗಾತ್ರ(vk_op_ಮೊತ್ತ_ಅಡ್ಡಸಾಲುಗಳು_ತಳ್ಳು_constants), {1, 1, 1}, { device->subgroup_ಗಾತ್ರ }, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cross_entropy_loss_f32, "cross_entropy_loss_f32", cross_entropy_loss_f32_len, cross_entropy_loss_f32_ದತ್ತಾಂಶ, "main", 3, ಗಾತ್ರ(vk_op_ತಳ್ಳು_constants), {1, 1, 1}, { device->subgroup_ಗಾತ್ರ }, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cross_entropy_loss_f32_wg512, "cross_entropy_loss_f32_wg512", cross_entropy_loss_f32_len, cross_entropy_loss_f32_ದತ್ತಾಂಶ, "main", 3, ಗಾತ್ರ(vk_op_ತಳ್ಳು_constants), {1, 1, 1}, { 512 }, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cross_entropy_loss_ಹಿಂಭಾಗ_f32, "cross_entropy_loss_back_f32", cross_entropy_loss_ಹಿಂಭಾಗ_f32_len, cross_entropy_loss_ಹಿಂಭಾಗ_f32_ದತ್ತಾಂಶ, "main", 4, ಗಾತ್ರ(vk_op_ತಳ್ಳು_constants), {1, 1, 1}, { device->subgroup_ಗಾತ್ರ }, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cross_entropy_loss_ಹಿಂಭಾಗ_f32_wg512, "cross_entropy_loss_back_f32_wg512", cross_entropy_loss_ಹಿಂಭಾಗ_f32_len, cross_entropy_loss_ಹಿಂಭಾಗ_f32_ದತ್ತಾಂಶ, "main", 4, ಗಾತ್ರ(vk_op_ತಳ್ಳು_constants), {1, 1, 1}, { 512 }, 1);
    // Intel Windows driver in range [32.0.101.8509, 32.0.101.8860) will crash when using fwht kernels so we gate that here
    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ can_use_fwht = device->driver_id != vk::DriverId::eIntelProprietaryWindows ||
        !ggml_vk_intel_windows_driver_ಒಳ_range(device->properties.driver_ಆವೃತ್ತಿ_2, 101, 8509, 101, 8860);
    ಆದರೆ (can_use_fwht && device->subgroup_basic && device->subgroup_shuffle) {
        ಪೂರ್ಣಾಂಕ idx = 0;
        ಗಾಗಿ (uint32_t n : {64, 128, 256, 512}) {
            ಆದರೆ (device->subgroup_ಗಾತ್ರ <= n) {
                ggml_vk_create_pipeline(device, device->pipeline_fwht_f32[idx], "fwht_f32", fwht_f32_len, fwht_f32_ದತ್ತಾಂಶ, "main", 2, ಗಾತ್ರ(vk_op_fwht_ತಳ್ಳು_constants), {1, 1, 1}, { device->subgroup_ಗಾತ್ರ, n }, 1, ನಿಜ, ನಿಜ, device->subgroup_ಗಾತ್ರ);
            }
            ++idx;
        }
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (can_use_fwht) {
        ಪೂರ್ಣಾಂಕ idx = 0;
        ಗಾಗಿ (uint32_t n : {64, 128, 256, 512}) {
            ಸ್ಥಿರಾಂಕ uint32_t ಖಂಡ_ಗಾತ್ರ_3 = std::min(device->subgroup_ಗಾತ್ರ, n);
            ggml_vk_create_pipeline(device, device->pipeline_fwht_f32[idx], "fwht_shmem_f32", fwht_shmem_f32_len, fwht_shmem_f32_ದತ್ತಾಂಶ, "main", 2, ಗಾತ್ರ(vk_op_fwht_ತಳ್ಳು_constants), {1, 1, 1}, { ಖಂಡ_ಗಾತ್ರ_3, n }, 1);
            ++idx;
        }
    }

    ಸ್ಥಿರಾಂಕ uint32_t cumsum_elem_ಪ್ರತಿ_thread = (device->vendor_id == VK_VENDOR_ID_AMD || device->vendor_id == VK_VENDOR_ID_INTEL) ? 2 : 4;
    ggml_vk_create_pipeline(device, device->pipeline_cumsum_f32,       "cumsum_f32", cumsum_f32_len, cumsum_f32_ದತ್ತಾಂಶ, "main", 2, ಗಾತ್ರ(vk_op_ಮೊತ್ತ_ಅಡ್ಡಸಾಲುಗಳು_ತಳ್ಳು_constants), {1, 1, 1}, { 256, device->subgroup_ಗಾತ್ರ, cumsum_elem_ಪ್ರತಿ_thread }, 1, ನಿಜ, ನಿಜ, device->subgroup_ಗಾತ್ರ);
    ggml_vk_create_pipeline(device, device->pipeline_cumsum_small_f32, "cumsum_f32", cumsum_f32_len, cumsum_f32_ದತ್ತಾಂಶ, "main", 2, ಗಾತ್ರ(vk_op_ಮೊತ್ತ_ಅಡ್ಡಸಾಲುಗಳು_ತಳ್ಳು_constants), {1, 1, 1}, { 128, device->subgroup_ಗಾತ್ರ, 1 }, 1, ನಿಜ, ನಿಜ, device->subgroup_ಗಾತ್ರ);
    ggml_vk_create_pipeline(device, device->pipeline_cumsum_multipass1_f32, "cumsum_multipass1_f32", cumsum_multipass1_f32_len, cumsum_multipass1_f32_ದತ್ತಾಂಶ, "main", 3, ಗಾತ್ರ(vk_op_ಮೊತ್ತ_ಅಡ್ಡಸಾಲುಗಳು_ತಳ್ಳು_constants), {256, 1, 1}, { 256, device->subgroup_ಗಾತ್ರ }, 1, ನಿಜ, ನಿಜ, device->subgroup_ಗಾತ್ರ);
    ggml_vk_create_pipeline(device, device->pipeline_cumsum_multipass2_f32, "cumsum_multipass2_f32", cumsum_multipass2_f32_len, cumsum_multipass2_f32_ದತ್ತಾಂಶ, "main", 3, ಗಾತ್ರ(vk_op_ಮೊತ್ತ_ಅಡ್ಡಸಾಲುಗಳು_ತಳ್ಳು_constants), {256, 1, 1}, { 256, device->subgroup_ಗಾತ್ರ }, 1, ನಿಜ, ನಿಜ, device->subgroup_ಗಾತ್ರ);

    ggml_vk_create_pipeline(device, device->pipeline_ಎಣಿಕೆ_equal_i32, "count_equal_i32", ಎಣಿಕೆ_equal_i32_len, ಎಣಿಕೆ_equal_i32_ದತ್ತಾಂಶ, "main", 3, ಗಾತ್ರ(vk_op_ತಳ್ಳು_constants), {512, 1, 1}, { device->subgroup_ಗಾತ್ರ }, 1);

    ಆದರೆ (device->subgroup_arithmetic && device->subgroup_vote && device->subgroup_require_ತುಂಬಿದ_support) {
        ggml_vk_create_pipeline(device, device->pipeline_ಎಣಿಕೆ_experts, "count_experts", ಎಣಿಕೆ_experts_subgroup_len, ಎಣಿಕೆ_experts_subgroup_ದತ್ತಾಂಶ, "main", 2, ಗಾತ್ರ(vk_op_ಎಣಿಕೆ_experts_ತಳ್ಳು_constants), {1, 1, 1}, {}, 1, ನಿಜ, ನಿಜ);
    } ಇಲ್ಲದಿದ್ದರೆ {
        ggml_vk_create_pipeline(device, device->pipeline_ಎಣಿಕೆ_experts, "count_experts", ಎಣಿಕೆ_experts_len, ಎಣಿಕೆ_experts_ದತ್ತಾಂಶ, "main", 2, ಗಾತ್ರ(vk_op_ಎಣಿಕೆ_experts_ತಳ್ಳು_constants), {1, 1, 1}, {}, 1, ನಿಜ);
    }

    // comb holds a token's 4x4 matrix in one 16-lane slice of a subgroup, so it
    // needs at least 16 lanes, pinned to a known size.
    ಆದರೆ (device->subgroup_basic && device->subgroup_shuffle && device->subgroup_require_ತುಂಬಿದ_support && device->subgroup_ಗಾತ್ರ >= 16) {
        ಸ್ಥಿರಾಂಕ uint32_t ಟೋಕನುಗಳು_ಪ್ರತಿ_workgroup = 4 * (device->subgroup_ಗಾತ್ರ / 16);
        ggml_vk_create_pipeline(device, device->pipeline_dsv4_hc_comb_f32, "dsv4_hc_comb_f32", dsv4_hc_comb_f32_len, dsv4_hc_comb_f32_ದತ್ತಾಂಶ, "main", 4, ಗಾತ್ರ(vk_op_dsv4_hc_comb_ತಳ್ಳು_constants), {ಟೋಕನುಗಳು_ಪ್ರತಿ_workgroup, 1, 1}, { device->subgroup_ಗಾತ್ರ }, 1, ನಿಜ, ನಿಜ, device->subgroup_ಗಾತ್ರ);
    }

    ggml_vk_create_pipeline(device, device->pipeline_dsv4_hc_pre_f32,        "dsv4_hc_pre_f32",        dsv4_hc_pre_f32_len,  dsv4_hc_pre_f32_ದತ್ತಾಂಶ,  "main", 3, ಗಾತ್ರ(vk_op_dsv4_hc_pre_ತಳ್ಳು_constants),  {256, 1, 1}, { 256, 0 }, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dsv4_hc_pre_gated_f32,  "dsv4_hc_pre_gated_f32",  dsv4_hc_pre_f32_len,  dsv4_hc_pre_f32_ದತ್ತಾಂಶ,  "main", 3, ಗಾತ್ರ(vk_op_dsv4_hc_pre_ತಳ್ಳು_constants),  {256, 1, 1}, { 256, 1 }, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dsv4_hc_post_f32,       "dsv4_hc_post_f32",       dsv4_hc_post_f32_len, dsv4_hc_post_f32_ದತ್ತಾಂಶ, "main", 5, ಗಾತ್ರ(vk_op_dsv4_hc_post_ತಳ್ಳು_constants), {256, 1, 1}, { 256, 1 }, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dsv4_hc_post_nocomb_f32,"dsv4_hc_post_nocomb_f32",dsv4_hc_post_f32_len, dsv4_hc_post_f32_ದತ್ತಾಂಶ, "main", 5, ಗಾತ್ರ(vk_op_dsv4_hc_post_ತಳ್ಳು_constants), {256, 1, 1}, { 256, 0 }, 1);

    ಗಾಗಿ (ಸ್ವಯಂಚಾಲಿತ &s : device->pipeline_solve_tri_f32) {
        ಸ್ಥಿರಾಂಕ vk_solve_tri_pipeline_state &state = s.ಮೊದಲ;

        // Max number of rows to load at a time, limited by shared memory
        ಸ್ಥಿರಾಂಕ uint32_t batch_N = device->properties.limits.max_Compute_Shared_ಸ್ಮರಣೆ_ಗಾತ್ರ / ((state.N + state.K) * ಗಾತ್ರ(ದಶಮಾಂಶ));
        // Need at least K invocations, and prefer a minimum of 128 to spread out loading shared memory
        ಸ್ಥಿರಾಂಕ uint32_t ಖಂಡ_ಗಾತ್ರ_3 = std::max(128u, 1u << (uint32_t)ceilf(log2f(ದಶಮಾಂಶ(state.K))));

        ggml_vk_create_pipeline(
            device, s.second, "solve_tri_f32",
            solve_tri_f32_len, solve_tri_f32_ದತ್ತಾಂಶ, "main", 3,
            ಗಾತ್ರ(vk_op_ದ್ವಿಮಾನ_ತಳ್ಳು_constants), {1, 1, 1}, { 0, state.N, state.K, batch_N, ಖಂಡ_ಗಾತ್ರ_3 }, 1, ನಿಜ);
    }

#ವ್ಯಾಖ್ಯಾನಿಸು IM2_ಲಂಬಸಾಲು(bda) \
    ggml_vk_create_pipeline(device, device->pipeline_im2col_f32, "im2col_f32", im2col_f32 ## bda ## _len, im2col_f32 ## bda ## ದತ್ತಾಂಶ_2, "main", 2, ಗಾತ್ರ(vk_op_im2col_ತಳ್ಳು_constants), {512, 1, 1}, { device->subgroup_ಗಾತ್ರ }, 1, ನಿಜ);   \
    ggml_vk_create_pipeline(device, device->pipeline_im2col_3d_f32, "im2col_3d_f32", im2col_3d_f32 ## bda ## _len, im2col_3d_f32 ## bda ## ದತ್ತಾಂಶ_2, "main", 2, ಗಾತ್ರ(vk_op_im2col_3d_ತಳ್ಳು_constants), {512, 1, 1}, { 512 }, 1, ನಿಜ);      \
    ggml_vk_create_pipeline(device, device->pipeline_im2col_f32_f16, "im2col_f32_f16", im2col_f32_f16 ## bda ## _len, im2col_f32_f16 ## bda ## ದತ್ತಾಂಶ_2, "main", 2, ಗಾತ್ರ(vk_op_im2col_ತಳ್ಳು_constants), {512, 1, 1}, { device->subgroup_ಗಾತ್ರ }, 1, ನಿಜ);   \
    ggml_vk_create_pipeline(device, device->pipeline_im2col_3d_f32_f16, "im2col_3d_f32_f16", im2col_3d_f32_f16 ## bda ## _len, im2col_3d_f32_f16 ## bda ## ದತ್ತಾಂಶ_2, "main", 2, ಗಾತ್ರ(vk_op_im2col_3d_ತಳ್ಳು_constants), {512, 1, 1}, { 512 }, 1, ನಿಜ);
    ಆದರೆ (device->shader_int64 && device->ಬಫರ್_device_ವಿಳಾಸ) {
        IM2_ಲಂಬಸಾಲು(_bda)
    } ಇಲ್ಲದಿದ್ದರೆ {
        IM2_ಲಂಬಸಾಲು()
    }

    ggml_vk_create_pipeline(device, device->pipeline_timestep_embedding_f32, "timestep_embedding_f32", timestep_embedding_f32_len, timestep_embedding_f32_ದತ್ತಾಂಶ, "main", 2, ಗಾತ್ರ(vk_op_timestep_embedding_ತಳ್ಳು_constants), {256, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_conv_transpose_1d_f32, "conv_transpose_1d_f32", conv_transpose_1d_f32_len, conv_transpose_1d_f32_ದತ್ತಾಂಶ, "main", 3, ಗಾತ್ರ(vk_op_conv_transpose_1d_ತಳ್ಳು_constants), {1, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_col2im_1d_f32,  "col2im_1d_f32",  col2im_1d_f32_len,  col2im_1d_f32_ದತ್ತಾಂಶ,  "main", 2, ಗಾತ್ರ(vk_op_col2im_1d_ತಳ್ಳು_constants), {256, 1, 1}, {}, 1, ನಿಜ);
    ggml_vk_create_pipeline(device, device->pipeline_col2im_1d_f16,  "col2im_1d_f16",  col2im_1d_f16_len,  col2im_1d_f16_ದತ್ತಾಂಶ,  "main", 2, ಗಾತ್ರ(vk_op_col2im_1d_ತಳ್ಳು_constants), {256, 1, 1}, {}, 1, ನಿಜ);
    ggml_vk_create_pipeline(device, device->pipeline_col2im_1d_bf16, "col2im_1d_bf16", col2im_1d_bf16_len, col2im_1d_bf16_ದತ್ತಾಂಶ, "main", 2, ಗಾತ್ರ(vk_op_col2im_1d_ತಳ್ಳು_constants), {256, 1, 1}, {}, 1, ನಿಜ);

    ggml_vk_create_pipeline(device, device->pipeline_ಹೊರಹೋಗು_prod_f32, "out_prod_f32", ಹೊರಹೋಗು_prod_f32_len, ಹೊರಹೋಗು_prod_f32_ದತ್ತಾಂಶ, "main", 3, ಗಾತ್ರ(vk_op_ದ್ವಿಮಾನ_ತಳ್ಳು_constants), {256, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_snake_f32,  "snake_f32",  snake_f32_len,  snake_f32_ದತ್ತಾಂಶ,  "main", 4, ಗಾತ್ರ(vk_op_snake_ತಳ್ಳು_constants), {256, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_snake_f16,  "snake_f16",  snake_f16_len,  snake_f16_ದತ್ತಾಂಶ,  "main", 4, ಗಾತ್ರ(vk_op_snake_ತಳ್ಳು_constants), {256, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_snake_bf16, "snake_bf16", snake_bf16_len, snake_bf16_ದತ್ತಾಂಶ, "main", 4, ಗಾತ್ರ(vk_op_snake_ತಳ್ಳು_constants), {256, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_pool1d_f32, "pool1d_f32", pool1d_f32_len, pool1d_f32_ದತ್ತಾಂಶ, "main", 2, ಗಾತ್ರ(vk_op_pool1d_ತಳ್ಳು_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_pool2d_f32, "pool2d_f32", pool2d_f32_len, pool2d_f32_ದತ್ತಾಂಶ, "main", 2, ಗಾತ್ರ(vk_op_pool2d_ತಳ್ಳು_constants), {1, 1, 1}, {512}, 1);
    ಆದರೆ (device->subgroup_arithmetic && device->subgroup_require_ತುಂಬಿದ_support) {
        ggml_vk_create_pipeline(device, device->pipeline_pool2d_subgroup_f32, "pool2d_subgroup_f32", pool2d_subgroup_f32_len, pool2d_subgroup_f32_ದತ್ತಾಂಶ, "main", 2, ಗಾತ್ರ(vk_op_pool2d_ತಳ್ಳು_constants), {1, 1, 1}, {128}, 1, ಸುಳ್ಳು, ನಿಜ, device->subgroup_ಗಾತ್ರ);
    }

    ggml_vk_create_pipeline(device, device->pipeline_rwkv_wkv6_f32, "rwkv_wkv6_f32", rwkv_wkv6_f32_len, rwkv_wkv6_f32_ದತ್ತಾಂಶ, "main", 7, ಗಾತ್ರ(vk_op_rwkv_wkv6_ತಳ್ಳು_constants), {1, 1, 1}, {device->subgroup_ಗಾತ್ರ}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_rwkv_wkv7_f32, "rwkv_wkv7_f32", rwkv_wkv7_f32_len, rwkv_wkv7_f32_ದತ್ತಾಂಶ, "main", 8, ಗಾತ್ರ(vk_op_rwkv_wkv7_ತಳ್ಳು_constants), {1, 1, 1}, {device->subgroup_ಗಾತ್ರ}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_gated_ರೇಖೀಯ_attn_f32, "gated_linear_attn_f32", gated_ರೇಖೀಯ_attn_f32_len, gated_ರೇಖೀಯ_attn_f32_ದತ್ತಾಂಶ, "main", 6, ಗಾತ್ರ(vk_op_gated_ರೇಖೀಯ_attn_ತಳ್ಳು_constants), {1, 1, 1}, {}, 1);

    ಗಾಗಿ (ggml_type k_type : lightning_indexer_k_types) {
        ಸ್ಥಿರಾಂಕ std::string name = "lightning_indexer_" + std::string(ggml_type_name(k_type)) + "_k_f32";
        ggml_vk_create_pipeline(device, device->pipeline_lightning_indexer_f32[k_type], name.c_str(), lightning_indexer_f32_len, lightning_indexer_f32_ದತ್ತಾಂಶ, "main", 5, ಗಾತ್ರ(vk_op_lightning_indexer_ತಳ್ಳು_constants), {1, 1, 1}, {(uint32_t)k_type, fa_ಖಂಡ_ಬೈಟುಗಳು(k_type)}, 1, ನಿಜ);
    }

    {
        ಸ್ಥಿರಾಂಕ uint32_t gdn_sizes[] = {16, 32, 64, 128};
        ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * gdn_ಹೆಸರುಗಳು[][2] = {
            {"gated_delta_net_f32_d16",     "gated_delta_net_f32_d16_kda"},
            {"gated_delta_net_f32_d32",     "gated_delta_net_f32_d32_kda"},
            {"gated_delta_net_f32_d64",     "gated_delta_net_f32_d64_kda"},
            {"gated_delta_net_f32_d128",    "gated_delta_net_f32_d128_kda"},
        };
        ಗಾಗಿ (uint32_t si = 0; si < 4; si++) {
            ಸ್ಥಿರಾಂಕ uint32_t S_V = gdn_sizes[si];
            GGML_ASSERT(is_pow2(S_V));

            // Intel Xe regresses at SIMD32 for this scan; prefer a narrower subgroup.
            uint32_t gdn_subgroup_ಗಾತ್ರ = device->subgroup_ಗಾತ್ರ;
            ಆದರೆ (device->vendor_id == VK_VENDOR_ID_INTEL && device->subgroup_ಗಾತ್ರ_control &&
                device->subgroup_min_ಗಾತ್ರ <= 16u && device->subgroup_max_ಗಾತ್ರ >= 16u) {
                gdn_subgroup_ಗಾತ್ರ = 16u;
            }
            uint32_t lanes_ಪ್ರತಿ_ಸ್ತಂಭ;
            ಆದರೆ (device->vendor_id == VK_VENDOR_ID_INTEL) {
                // Intel Xe: full-width reduction (min rows/lane) ~10x over the COLS_PER_WG=8 rule.
                lanes_ಪ್ರತಿ_ಸ್ತಂಭ = std::min(gdn_subgroup_ಗಾತ್ರ, S_V);
            } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (S_V >= 128u && device->subgroup_clustered) {
                // COLS_PER_WG=8: measured optimum (Ampere sg32->4, Vega20 sg64->8).
                lanes_ಪ್ರತಿ_ಸ್ತಂಭ = std::max(1u, gdn_subgroup_ಗಾತ್ರ / 8u);
            } ಇಲ್ಲದಿದ್ದರೆ {
                // Use largest power-of-two that divides both S_V and subgroup_size so that
                // (1) S_V % lanes_per_column == 0 and (2) S_V % (subgroup_size / lanes_per_column) == 0.
                // This means we don't need extra bounds checking logic in the shader.
                lanes_ಪ್ರತಿ_ಸ್ತಂಭ = std::min(S_V, gdn_subgroup_ಗಾತ್ರ);
            }

            // gated_delta_net.comp relies on S_V % COLS_PER_WG == 0 and
            // S_V % LANES_PER_COLUMN == 0 to avoid bounds checks.
            ಆಗ (lanes_ಪ್ರತಿ_ಸ್ತಂಭ > 1u) {
                ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ valid_lanes = (gdn_subgroup_ಗಾತ್ರ % lanes_ಪ್ರತಿ_ಸ್ತಂಭ) == 0 &&
                                         (S_V % lanes_ಪ್ರತಿ_ಸ್ತಂಭ) == 0;
                ಸ್ಥಿರಾಂಕ uint32_t ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ_ಪ್ರತಿ_wg = valid_lanes ? gdn_subgroup_ಗಾತ್ರ / lanes_ಪ್ರತಿ_ಸ್ತಂಭ : 0;
                ಆದರೆ (valid_lanes && ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ_ಪ್ರತಿ_wg > 0 && (S_V % ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ_ಪ್ರತಿ_wg) == 0) {
                    ನಿಲ್ಲಿಸು;
                }
                lanes_ಪ್ರತಿ_ಸ್ತಂಭ >>= 1u;
            }

            GGML_ASSERT((gdn_subgroup_ಗಾತ್ರ % lanes_ಪ್ರತಿ_ಸ್ತಂಭ) == 0);
            GGML_ASSERT((S_V % lanes_ಪ್ರತಿ_ಸ್ತಂಭ) == 0);
            GGML_ASSERT((S_V % (gdn_subgroup_ಗಾತ್ರ / lanes_ಪ್ರತಿ_ಸ್ತಂಭ)) == 0);

            ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ need_partial_subgroup_reduce = lanes_ಪ್ರತಿ_ಸ್ತಂಭ != 1u && lanes_ಪ್ರತಿ_ಸ್ತಂಭ < gdn_subgroup_ಗಾತ್ರ;
            ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ use_clustered_reduce = device->subgroup_arithmetic && device->subgroup_clustered && need_partial_subgroup_reduce;
            ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ use_subgroup_reduce = device->subgroup_arithmetic && !need_partial_subgroup_reduce;
            ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ use_subgroup_ops = use_clustered_reduce || use_subgroup_reduce;
            size_t gdn_len;
            ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * gdn_ದತ್ತಾಂಶ;
            ಆದರೆ (use_clustered_reduce) {
                gdn_len = gated_delta_net_f32_len;
                gdn_ದತ್ತಾಂಶ = (ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *)gated_delta_net_f32_ದತ್ತಾಂಶ;
            } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (use_subgroup_reduce) {
                gdn_len = gated_delta_net_f32_nocluster_len;
                gdn_ದತ್ತಾಂಶ = (ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *)gated_delta_net_f32_nocluster_ದತ್ತಾಂಶ;
            } ಇಲ್ಲದಿದ್ದರೆ {
                gdn_len = gated_delta_net_f32_shmem_len;
                gdn_ದತ್ತಾಂಶ = (ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *)gated_delta_net_f32_shmem_ದತ್ತಾಂಶ;
            }

            ಸ್ಥಿರಾಂಕ uint32_t ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ_ಪ್ರತಿ_wg = gdn_subgroup_ಗಾತ್ರ / lanes_ಪ್ರತಿ_ಸ್ತಂಭ;
            ಸ್ಥಿರಾಂಕ std::array<uint32_t, 3> wg_denoms = {1u, 1u, ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ_ಪ್ರತಿ_wg};

            ಗಾಗಿ (uint32_t kda = 0; kda < 2; kda++) {
                ggml_vk_create_pipeline(device, device->pipeline_gated_delta_net[si][kda],
                    gdn_ಹೆಸರುಗಳು[si][kda], gdn_len, gdn_ದತ್ತಾಂಶ, "main", 7, ಗಾತ್ರ(vk_op_gated_delta_net_ತಳ್ಳು_constants),
                    wg_denoms, {S_V, kda, gdn_subgroup_ಗಾತ್ರ, lanes_ಪ್ರತಿ_ಸ್ತಂಭ}, 1, ನಿಜ, use_subgroup_ops, gdn_subgroup_ಗಾತ್ರ);
            }
        }
    }

    ಆದರೆ (device->subgroup_arithmetic && device->subgroup_require_ತುಂಬಿದ_support) {
        ggml_vk_create_pipeline(device, device->pipeline_ssm_ಪರಿಶೀಲನೆ_f32_d128, "ssm_scan_128_f32", ssm_ಪರಿಶೀಲನೆ_subgroup_f32_len, ssm_ಪರಿಶೀಲನೆ_subgroup_f32_ದತ್ತಾಂಶ, "main", 8, ಗಾತ್ರ(vk_op_ssm_ಪರಿಶೀಲನೆ_ತಳ್ಳು_constants), {1, 1, 1}, {128, device->subgroup_ಗಾತ್ರ}, 1, ನಿಜ, ನಿಜ);
        ggml_vk_create_pipeline(device, device->pipeline_ssm_ಪರಿಶೀಲನೆ_f32_d256, "ssm_scan_256_f32", ssm_ಪರಿಶೀಲನೆ_subgroup_f32_len, ssm_ಪರಿಶೀಲನೆ_subgroup_f32_ದತ್ತಾಂಶ, "main", 8, ಗಾತ್ರ(vk_op_ssm_ಪರಿಶೀಲನೆ_ತಳ್ಳು_constants), {1, 1, 1}, {256, device->subgroup_ಗಾತ್ರ}, 1, ನಿಜ, ನಿಜ);
    } ಇಲ್ಲದಿದ್ದರೆ {
        ggml_vk_create_pipeline(device, device->pipeline_ssm_ಪರಿಶೀಲನೆ_f32_d128, "ssm_scan_128_f32", ssm_ಪರಿಶೀಲನೆ_f32_len, ssm_ಪರಿಶೀಲನೆ_f32_ದತ್ತಾಂಶ, "main", 8, ಗಾತ್ರ(vk_op_ssm_ಪರಿಶೀಲನೆ_ತಳ್ಳು_constants), {1, 1, 1}, {128, device->subgroup_ಗಾತ್ರ, 16}, 1, ನಿಜ, ನಿಜ);
        ggml_vk_create_pipeline(device, device->pipeline_ssm_ಪರಿಶೀಲನೆ_f32_d256, "ssm_scan_256_f32", ssm_ಪರಿಶೀಲನೆ_f32_len, ssm_ಪರಿಶೀಲನೆ_f32_ದತ್ತಾಂಶ, "main", 8, ಗಾತ್ರ(vk_op_ssm_ಪರಿಶೀಲನೆ_ತಳ್ಳು_constants), {1, 1, 1}, {256, device->subgroup_ಗಾತ್ರ, 16}, 1, ನಿಜ, ನಿಜ);
    }

    ggml_vk_create_pipeline(device, device->pipeline_ssm_conv_f32,           "ssm_conv_f32",           ssm_conv_f32_len, ssm_conv_f32_ದತ್ತಾಂಶ, "main", 4, ಗಾತ್ರ(vk_op_ssm_conv_ತಳ್ಳು_constants), {32, 16, 1}, {32, 16, 0, 0}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_ssm_conv_silu_f32,      "ssm_conv_silu_f32",      ssm_conv_f32_len, ssm_conv_f32_ದತ್ತಾಂಶ, "main", 4, ಗಾತ್ರ(vk_op_ssm_conv_ತಳ್ಳು_constants), {32, 16, 1}, {32, 16, 0, 1}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_ssm_conv_bias_silu_f32, "ssm_conv_bias_silu_f32", ssm_conv_f32_len, ssm_conv_f32_ದತ್ತಾಂಶ, "main", 4, ಗಾತ್ರ(vk_op_ssm_conv_ತಳ್ಳು_constants), {32, 16, 1}, {32, 16, 1, 1}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_opt_ಹೆಜ್ಜೆ_adamw_f32, "opt_step_adamw_f32", opt_ಹೆಜ್ಜೆ_adamw_f32_len, opt_ಹೆಜ್ಜೆ_adamw_f32_ದತ್ತಾಂಶ, "main", 5, ಗಾತ್ರ(vk_op_ತಳ್ಳು_constants), {512, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_opt_ಹೆಜ್ಜೆ_sgd_f32, "opt_step_sgd_f32", opt_ಹೆಜ್ಜೆ_sgd_f32_len, opt_ಹೆಜ್ಜೆ_sgd_f32_ದತ್ತಾಂಶ, "main", 3, ಗಾತ್ರ(vk_op_ತಳ್ಳು_constants), {512, 1, 1}, {}, 1);

    // conv2d, conv_transpose_2d, conv3d
    ಗಾಗಿ (uint32_t s = 0; s < CONV_SHAPE_ಎಣಿಕೆ; ++s) {
        // smaller WG for the small-tile fallback gives more concurrent WGs per SM
        uint32_t conv2d_WG_ಗಾತ್ರ  = (s == CONV_SHAPE_64x32) ? 128 : 256;
        uint32_t use_collectives = 0;  // Enables subgroup ops for preventing the re-calculation of indices.
        uint32_t conv2d_TS_K     = (s == CONV_SHAPE_64x32) ? 4 : 8;
        uint32_t conv2d_SHMEM_PAD = 4;
        vk_conv_ಖಂಡ_ಗಾತ್ರ conv2d_BS = vk_conv_ಖಂಡ_sizes[s];
        ತಾರ್ಕಿಕ conv2d_UNROLL = ನಿಜ;

#ಆದರೆ defined(GGML_VULKAN_COOPMAT2_GLSLC_SUPPORT)
        ಆದರೆ (device->coopmat2) {
            conv2d_SHMEM_PAD = 8; // 8 float16_t
        }
#ಆದರೆ_ಅಂತ್ಯ

        ಆದರೆ (device->vendor_id == VK_VENDOR_ID_INTEL) {
            conv2d_SHMEM_PAD = 0;
            conv2d_UNROLL = ಸುಳ್ಳು;
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (device->vendor_id == VK_VENDOR_ID_AMD) {
            conv2d_SHMEM_PAD = device->architecture == vk_device_architecture::AMD_GCN ? 1 : 4;
            ಆದರೆ (s == CONV_SHAPE_128x128 && device->architecture != vk_device_architecture::AMD_GCN) {
                conv2d_UNROLL = ಸುಳ್ಳು;
            }
        }

        // Use collectives on pre-Turing NVIDIA GPUs and GCN AMD cards, which had slower integer math.
        ತಾರ್ಕಿಕ allow_collectives_nv = device->vendor_id != VK_VENDOR_ID_NVIDIA ||
                                    device->architecture == vk_device_architecture::NVIDIA_PRE_TURING;
        ತಾರ್ಕಿಕ allow_collectives_amd = device->vendor_id != VK_VENDOR_ID_AMD ||
                                     device->architecture == vk_device_architecture::AMD_GCN;

        ಆದರೆ (device->subgroup_shuffle &&
            device->vendor_id != VK_VENDOR_ID_INTEL &&   // Do not enable collectives on Intel, see PR 14316.
            allow_collectives_nv &&
            allow_collectives_amd) {
            use_collectives = 1;
            conv2d_BS.CRS   = std::min(
                device->subgroup_ಗಾತ್ರ,
                conv2d_BS.CRS);  // CRS block size should be capped at subgroup size for correctness when shuffle is used.
        }

        // cm1 is used only when cm2 is unavailable; capped at 64x128 (due to shared memory size).
        // Requires 16x16x16 f16-acc since that's the fragment shape hard-coded in the shader.
        // Subgroup size must be 32 or 64 (to keep WG_SIZE sane) and we need
        // subgroup_size_control to force the driver to actually use it.
        ತಾರ್ಕಿಕ conv2d_use_cm1 = ಸುಳ್ಳು;
#ಆದರೆ defined(VK_KHR_cooperative_ಮಾತೃಕೆ) && defined(GGML_VULKAN_COOPMAT_GLSLC_SUPPORT)
        conv2d_use_cm1 = !device->coopmat2 &&
                         device->coopmat_support && device->coopmat_support_16x16x16_f16acc &&
                         device->subgroup_ಗಾತ್ರ_control &&
                         (device->subgroup_ಗಾತ್ರ == 32 || device->subgroup_ಗಾತ್ರ == 64) &&
                         s != CONV_SHAPE_128x128;
#ಆದರೆ_ಅಂತ್ಯ

        ಸ್ಥಿರಾಂಕ uint32_t conv2d_cm1_shmem_pad = 8;

        ಸ್ವಯಂಚಾಲಿತ shmem_req = [&](uint32_t pad, ತಾರ್ಕಿಕ csh_store, ತಾರ್ಕಿಕ fp16_shmem) {
            ಸ್ಥಿರಾಂಕ uint32_t elem_ಗಾತ್ರ = fp16_shmem ? (uint32_t)ಗಾತ್ರ(uint16_t) : (uint32_t)ಗಾತ್ರ(ದಶಮಾಂಶ);
            ಸ್ಥಿರಾಂಕ uint32_t csh_elems = csh_store ? conv2d_BS.K * conv2d_BS.NPQ : 0u;
            ಹಿಂತಿರುಗಿ (conv2d_BS.K * (conv2d_BS.CRS + pad) + conv2d_BS.CRS * (conv2d_BS.NPQ + pad) + csh_elems) * elem_ಗಾತ್ರ;
        };

        // 2D, transpose-2D, and 3D conv use the same KxCRS @ CRSxNPQ shmem
        // layout. cm1 needs Csh for output, so check before applying cm1 params.
        ಆದರೆ (conv2d_use_cm1 && device->properties.limits.max_Compute_Shared_ಸ್ಮರಣೆ_ಗಾತ್ರ < shmem_req(conv2d_cm1_shmem_pad, ನಿಜ, ನಿಜ)) {
            conv2d_use_cm1 = ಸುಳ್ಳು;
        }

        uint32_t conv2d_WM = 16, conv2d_WN = 16;  // cm1 subgroup tile, ignored otherwise
        ಆದರೆ (conv2d_use_cm1) {
            conv2d_SHMEM_PAD = conv2d_cm1_shmem_pad;
            // 16x16x16 fragments; pick WM/WN to keep WG_SIZE at 256
            // (i.e. 8 subgroups for sg=32, 4 subgroups for sg=64).
            ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ sg64 = (device->subgroup_ಗಾತ್ರ == 64);
            ಆಯ್ಕೆ (s) {
                ಸ್ಥಿತಿ CONV_SHAPE_64x32:   conv2d_WM = sg64 ? 32 : 16; conv2d_WN = 16; ನಿಲ್ಲಿಸು;
                ಸ್ಥಿತಿ CONV_SHAPE_64x128:  conv2d_WM = 32; conv2d_WN = sg64 ? 64 : 32; ನಿಲ್ಲಿಸು;
                ಸ್ಥಿತಿ CONV_SHAPE_32x256:  conv2d_WM = sg64 ? 16 : 32; conv2d_WN = sg64 ? 128 : 32; ನಿಲ್ಲಿಸು;
                ಪೂರ್ವನಿಯೋಜಿತ: ನಿಲ್ಲಿಸು;
            }
            ಸ್ಥಿರಾಂಕ uint32_t warps_M = conv2d_BS.K / conv2d_WM;
            ಸ್ಥಿರಾಂಕ uint32_t warps_N = conv2d_BS.NPQ / conv2d_WN;
            conv2d_WG_ಗಾತ್ರ         = warps_M * warps_N * device->subgroup_ಗಾತ್ರ;
        }

        // stage cm2 accumulator through shmem for coalesced global stores;
        // skipped on 128x128 where the extra Csh footprint hurts occupancy.
        // cm1 always uses the staged path.
        uint32_t conv2d_csh_store = (device->coopmat2 && s != CONV_SHAPE_128x128) ? 1u : 0u;
        ಆದರೆ (conv2d_use_cm1) {
            conv2d_csh_store = 1;
        }

        // shmem is fp16 on cm2/cm1 (matches Csh), fp32 on scalar
        ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ conv2d_use_fp16_shmem = device->coopmat2 || conv2d_use_cm1;

        // shrink CRS if the non-cm1 config still doesn't fit
        ಆದರೆ (device->properties.limits.max_Compute_Shared_ಸ್ಮರಣೆ_ಗಾತ್ರ < shmem_req(conv2d_SHMEM_PAD, conv2d_csh_store, conv2d_use_fp16_shmem)) {
            GGML_ASSERT(!conv2d_use_cm1);
            conv2d_BS.CRS = 8;
            ಆದರೆ (use_collectives) {
                conv2d_BS.CRS = std::min(device->subgroup_ಗಾತ್ರ, conv2d_BS.CRS);
            }
            conv2d_csh_store = 0;
        }

        std::array<uint32_t, 3> wg_denoms = { conv2d_BS.K, 1, 1 };
        std::vector<uint32_t> spec_constants = { conv2d_WG_ಗಾತ್ರ, conv2d_BS.K, conv2d_BS.CRS, conv2d_BS.NPQ, conv2d_TS_K, use_collectives, conv2d_SHMEM_PAD };

        // cm1 needs a fixed subgroup width to match the WG_SIZE we computed
        ಸ್ಥಿರಾಂಕ uint32_t conv2d_required_subgroup_ಗಾತ್ರ = conv2d_use_cm1 ? device->subgroup_ಗಾತ್ರ : 0;

#ವ್ಯಾಖ್ಯಾನಿಸು CREATE_CONV(name, type_suffix, spv_suffix) \
        ಗಾಗಿ (ಸ್ವಯಂಚಾಲಿತ &c : device->pipeline_##name##type_suffix[s]) { \
            ಸ್ಥಿರಾಂಕ vk_conv2d_pipeline_state &state = c.ಮೊದಲ;  \
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
                ಗಾತ್ರ(vk_op_conv2d_ತಳ್ಳು_constants), wg_denoms, spec_constants_cpy, 1, ನಿಜ, use_collectives || conv2d_required_subgroup_ಗಾತ್ರ, conv2d_required_subgroup_ಗಾತ್ರ);    \
        }
#ವ್ಯಾಖ್ಯಾನಿಸು CREATE_CONVS(spv_suffix) \
        CREATE_CONV(conv2d, _f32, spv_suffix) \
        CREATE_CONV(conv2d, _f16_f32, spv_suffix) \
        CREATE_CONV(conv_transpose_2d, _f32, spv_suffix) \
        CREATE_CONV(conv_transpose_2d, _f16_f32, spv_suffix)
#ಆದರೆ defined(GGML_VULKAN_COOPMAT2_GLSLC_SUPPORT)
        ಆದರೆ (device->coopmat2) {
            CREATE_CONVS(_cm2)
        } ಇಲ್ಲದಿದ್ದರೆ
#ಆದರೆ_ಅಂತ್ಯ
#ಆದರೆ defined(VK_KHR_cooperative_ಮಾತೃಕೆ) && defined(GGML_VULKAN_COOPMAT_GLSLC_SUPPORT)
        ಆದರೆ (conv2d_use_cm1) {
            CREATE_CONVS(_cm1)
        } ಇಲ್ಲದಿದ್ದರೆ
#ಆದರೆ_ಅಂತ್ಯ
        ಆದರೆ (conv2d_UNROLL) {
            CREATE_CONVS(_unroll)
        } ಇಲ್ಲದಿದ್ದರೆ {
            CREATE_CONVS( )
        }
#ವ್ಯಾಖ್ಯಾನ_ರದ್ದು CREATE_CONV
#ವ್ಯಾಖ್ಯಾನ_ರದ್ದು CREATE_CONVS

        std::vector<uint32_t> conv3d_spec_constants = { conv2d_WG_ಗಾತ್ರ, conv2d_BS.K, conv2d_BS.CRS, conv2d_BS.NPQ, conv2d_TS_K, conv2d_SHMEM_PAD };
#ವ್ಯಾಖ್ಯಾನಿಸು CREATE_CONV3D(type_suffix, spv_suffix) \
        ಗಾಗಿ (ಸ್ವಯಂಚಾಲಿತ &c : device->pipeline_conv3d##type_suffix[s]) { \
            ಸ್ಥಿರಾಂಕ vk_conv3d_pipeline_state &state = c.ಮೊದಲ; \
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
                ಗಾತ್ರ(vk_op_conv3d_ತಳ್ಳು_constants), wg_denoms, spec_constants_cpy, 1, ನಿಜ, conv2d_required_subgroup_ಗಾತ್ರ != 0, conv2d_required_subgroup_ಗಾತ್ರ); \
        }
#ಆದರೆ defined(GGML_VULKAN_COOPMAT2_GLSLC_SUPPORT)
        ಆದರೆ (device->coopmat2) {
            CREATE_CONV3D(_f32, _cm2)
            CREATE_CONV3D(_f16_f32, _cm2)
        } ಇಲ್ಲದಿದ್ದರೆ
#ಆದರೆ_ಅಂತ್ಯ
#ಆದರೆ defined(VK_KHR_cooperative_ಮಾತೃಕೆ) && defined(GGML_VULKAN_COOPMAT_GLSLC_SUPPORT)
        ಆದರೆ (conv2d_use_cm1) {
            CREATE_CONV3D(_f32, _cm1)
            CREATE_CONV3D(_f16_f32, _cm1)
        } ಇಲ್ಲದಿದ್ದರೆ
#ಆದರೆ_ಅಂತ್ಯ
        ಆದರೆ (conv2d_UNROLL) {
            CREATE_CONV3D(_f32, _unroll)
            CREATE_CONV3D(_f16_f32, _unroll)
        } ಇಲ್ಲದಿದ್ದರೆ {
            CREATE_CONV3D(_f32, )
            CREATE_CONV3D(_f16_f32, )
        }
#ವ್ಯಾಖ್ಯಾನ_ರದ್ದು CREATE_CONV3D
    }

    ggml_vk_create_pipeline(device, device->pipeline_conv2d_dw_whcn_f32, "conv2d_dw_whcn_f32", conv2d_dw_whcn_f32_len, conv2d_dw_whcn_f32_ದತ್ತಾಂಶ, "main", 3, ಗಾತ್ರ(vk_op_conv2d_dw_ತಳ್ಳು_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_conv2d_dw_cwhn_f32, "conv2d_dw_cwhn_f32", conv2d_dw_cwhn_f32_len, conv2d_dw_cwhn_f32_ದತ್ತಾಂಶ, "main", 3, ಗಾತ್ರ(vk_op_conv2d_dw_ತಳ್ಳು_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_conv2d_dw_whcn_f16_f32, "conv2d_dw_whcn_f16_f32", conv2d_dw_whcn_f16_f32_len, conv2d_dw_whcn_f16_f32_ದತ್ತಾಂಶ, "main", 3, ಗಾತ್ರ(vk_op_conv2d_dw_ತಳ್ಳು_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_conv2d_dw_cwhn_f16_f32, "conv2d_dw_cwhn_f16_f32", conv2d_dw_cwhn_f16_f32_len, conv2d_dw_cwhn_f16_f32_ದತ್ತಾಂಶ, "main", 3, ಗಾತ್ರ(vk_op_conv2d_dw_ತಳ್ಳು_constants), {512, 1, 1}, {}, 1);

    ಗಾಗಿ (uint32_t use_ತಳ್ಳು = 0; use_ತಳ್ಳು < 2; ++use_ತಳ್ಳು) {
        ಗಾಗಿ (uint32_t i = 0; i < ಅಂಕೆ_topk_moe_pipelines; ++i) {
            ggml_vk_create_pipeline2(device, device->pipeline_topk_moe[i][use_ತಳ್ಳು], "topk_moe_f32_"+std::to_string(i), topk_moe_f32_len, topk_moe_f32_ದತ್ತಾಂಶ, "main", 4, ಗಾತ್ರ(vk_op_topk_moe_ತಳ್ಳು_constants), {1, 1, 1}, {device->subgroup_ಗಾತ್ರ, 1u<<i, use_ತಳ್ಳು}, 1, ನಿಜ, ನಿಜ, device->subgroup_ಗಾತ್ರ);
        }
    }

    // Drop compile_mutex so other threads can walk while we compile.
    compile_ಬೀಗ.unlock();

    // Compile what we claimed; create_pipeline_func reacquires compile_mutex
    // at the end to flip compile_pending/compiled and notify waiters.
    ಆದರೆ (has_claimed_task) {
        ಸ್ವಯಂಚಾಲಿತ & task = claimed_task;
        ggml_vk_create_pipeline_func(device, task.pipeline, task.spv_ಗಾತ್ರ, task.spv_ದತ್ತಾಂಶ,
                                     task.entrypoint, task.parameter_ಎಣಿಕೆ, task.wg_denoms,
                                     task.specialization_constants, task.disable_robustness,
                                     task.require_ತುಂಬಿದ_subgroups, task.required_subgroup_ಗಾತ್ರ);
    }

    // Another thread may be compiling the pipeline we need; block on it here.
    ಆದರೆ (wait_pipeline) {
        std::unique_lock<std::mutex> wait_ಬೀಗ(device->compile_mutex);
        device->compile_cv.wait(wait_ಬೀಗ, [&] {
            ಹಿಂತಿರುಗಿ wait_pipeline->compiled.load();
        });
    }
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_vk_external_ಸ್ಮರಣೆ_host_usable(vk::DriverId driver_id) {
    // Disable external_memory_host until https://github.com/KhronosGroup/MoltenVK/pull/2622
    // is available in the Vulkan SDK.
    ಹಿಂತಿರುಗಿ driver_id != vk::DriverId::eMoltenvk;
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_vk_host_ಬಫರ್_import_usable(uint32_t vendor_id) {
    // amdgpu re-faults all pages of every imported host allocation on each submission
    ಹಿಂತಿರುಗಿ vendor_id != VK_VENDOR_ID_AMD;
}

vk_device ggml_vk_get_device(size_t idx) {
    VK_LOG_DEBUG("ggml_vk_get_device(" << idx << ")");

    ಸ್ಥಿರ std::mutex mutex;
    std::lock_guard<std::mutex> ಬೀಗ(mutex);

    ಆದರೆ (vk_instance.devices[idx] == ಶೂನ್ಯಸೂಚಕ) {
        VK_LOG_DEBUG("Initializing new vk_device");
        vk_device device = std::make_shared<vk_device_struct>();

        device->ಸ್ಮರಣೆ_logger = std::unique_ptr<vk_ಸ್ಮರಣೆ_logger>(ಹೊಸ vk_ಸ್ಮರಣೆ_logger());

        size_t dev_ಅಂಕೆ = vk_instance.device_indices[idx];

        std::vector<vk::PhysicalDevice> physical_devices = vk_instance.instance.enumeratePhysicalDevices();

        ಆದರೆ (dev_ಅಂಕೆ >= physical_devices.size()) {
            std::cerr << "ggml_vulkan: Device with index " << dev_ಅಂಕೆ << " does not exist." << std::ಸಾಲು_ಅಂತ್ಯ;
            ಎಸೆ std::runtime_error("Device not found");
        }

        device->physical_device = physical_devices[dev_ಅಂಕೆ];
        ಸ್ಥಿರಾಂಕ std::vector<vk::ExtensionProperties> ext_props = device->physical_device.enumerateDeviceExtensionProperties();

        device->architecture = get_device_architecture(device->physical_device);

        ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ* GGML_VK_PREFER_HOST_ಸ್ಮರಣೆ = getenv("GGML_VK_PREFER_HOST_MEMORY");
        device->prefer_host_ಸ್ಮರಣೆ = GGML_VK_PREFER_HOST_ಸ್ಮರಣೆ != ಶೂನ್ಯಸೂಚಕ;

        ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ* GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM = getenv("GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM");
        device->disable_host_visible_vidmem = GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM != ಶೂನ್ಯಸೂಚಕ;

        ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ* GGML_VK_ALLOW_SYSMEM_FALLBACK = getenv("GGML_VK_ALLOW_SYSMEM_FALLBACK");
        device->allow_sysmem_fallback = GGML_VK_ALLOW_SYSMEM_FALLBACK != ಶೂನ್ಯಸೂಚಕ;

        ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ* GGML_VK_DISABLE_ನಕ್ಷೆ_OPTIMIZE = getenv("GGML_VK_DISABLE_GRAPH_OPTIMIZE");
        device->disable_ನಕ್ಷೆ_optimize = GGML_VK_DISABLE_ನಕ್ಷೆ_OPTIMIZE != ಶೂನ್ಯಸೂಚಕ;

        ತಾರ್ಕಿಕ fp16_storage = ಸುಳ್ಳು;
        ತಾರ್ಕಿಕ fp16_compute = ಸುಳ್ಳು;
        ತಾರ್ಕಿಕ maintenance4_support = ಸುಳ್ಳು;
        ತಾರ್ಕಿಕ sm_builtins = ಸುಳ್ಳು;
        ತಾರ್ಕಿಕ amd_shader_core_properties2 = ಸುಳ್ಳು;
        ತಾರ್ಕಿಕ pipeline_robustness = ಸುಳ್ಳು;
        ತಾರ್ಕಿಕ coopmat2_support = ಸುಳ್ಳು;
        ತಾರ್ಕಿಕ coopmat2_decode_ಸದಿಶ_support = ಸುಳ್ಳು;
        ತಾರ್ಕಿಕ pipeline_executable_properties_support = ಸುಳ್ಳು;
        ತಾರ್ಕಿಕ internally_sync_support = ಸುಳ್ಳು;
        device->coopmat_support = ಸುಳ್ಳು;
        device->integer_dot_ಗುಣಲಬ್ಧ = ಸುಳ್ಳು;
        device->shader_64b_indexing = ಸುಳ್ಳು;
        ತಾರ್ಕಿಕ bfloat16_support = ಸುಳ್ಳು;
        ತಾರ್ಕಿಕ dot2_f16_support = ಸುಳ್ಳು;
        ತಾರ್ಕಿಕ ocp_microscaling_extension = ಸುಳ್ಳು;
        ತಾರ್ಕಿಕ shader_float8_extension = ಸುಳ್ಳು;

        ಗಾಗಿ (ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ& properties : ext_props) {
            ಆದರೆ (strcmp("VK_KHR_maintenance4", properties.extensionName) == 0) {
                maintenance4_support = ನಿಜ;
            } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (strcmp("VK_KHR_16bit_storage", properties.extensionName) == 0) {
                fp16_storage = ನಿಜ;
            } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (strcmp("VK_KHR_shader_float16_int8", properties.extensionName) == 0) {
                fp16_compute = ನಿಜ;
            } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (strcmp("VK_NV_shader_sm_builtins", properties.extensionName) == 0) {
                sm_builtins = ನಿಜ;
            } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (strcmp("VK_AMD_shader_core_properties2", properties.extensionName) == 0) {
                amd_shader_core_properties2 = ನಿಜ;
            } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (strcmp("VK_EXT_pipeline_robustness", properties.extensionName) == 0) {
                pipeline_robustness = ನಿಜ;
            } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (strcmp("VK_EXT_subgroup_size_control", properties.extensionName) == 0) {
                device->subgroup_ಗಾತ್ರ_control = ನಿಜ;
#ಆದರೆ defined(GGML_VULKAN_COOPMAT_GLSLC_SUPPORT)
            } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (strcmp("VK_KHR_cooperative_matrix", properties.extensionName) == 0 &&
                       !getenv("GGML_VK_DISABLE_COOPMAT")) {
                device->coopmat_support = ನಿಜ;
                device->coopmat_m = 0;
                device->coopmat_n = 0;
                device->coopmat_k = 0;
#ಆದರೆ_ಅಂತ್ಯ
#ಆದರೆ defined(GGML_VULKAN_COOPMAT2_GLSLC_SUPPORT)
            } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (strcmp("VK_NV_cooperative_matrix2", properties.extensionName) == 0 &&
                       !getenv("GGML_VK_DISABLE_COOPMAT2")) {
                coopmat2_support = ನಿಜ;
#ಆದರೆ_ಅಂತ್ಯ
            } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (strcmp(VK_NV_COOPERATIVE_ಮಾತೃಕೆ_DECODE_ಸದಿಶ_EXTENSION_NAME, properties.extensionName) == 0 &&
                       !getenv("GGML_VK_DISABLE_COOPMAT2_DECODE_VECTOR")) {
                coopmat2_decode_ಸದಿಶ_support = ನಿಜ;
#ಆದರೆ defined(GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT)
            } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (strcmp("VK_KHR_shader_integer_dot_product", properties.extensionName) == 0 &&
                       !getenv("GGML_VK_DISABLE_INTEGER_DOT_PRODUCT")) {
                device->integer_dot_ಗುಣಲಬ್ಧ = ನಿಜ;
#ಆದರೆ_ಅಂತ್ಯ
#ಆದರೆ defined(GGML_VULKAN_BFLOAT16_GLSLC_SUPPORT)
            } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (strcmp("VK_KHR_shader_bfloat16", properties.extensionName) == 0 &&
                       !getenv("GGML_VK_DISABLE_BFLOAT16")) {
                bfloat16_support = ನಿಜ;
#ಆದರೆ_ಅಂತ್ಯ
#ಆದರೆ defined(GGML_VULKAN_FLOAT_E2M1_GLSLC_SUPPORT)
            } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (strcmp(VK_EXT_SHADER_OCP_MICROSCALING_TYPES_EXTENSION_NAME, properties.extensionName) == 0) {
                ocp_microscaling_extension = ನಿಜ;
#ಆದರೆ_ಅಂತ್ಯ
#ಆದರೆ defined(GGML_VULKAN_FLOAT_E4M3_GLSLC_SUPPORT)
            } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (strcmp(VK_EXT_SHADER_FLOAT8_EXTENSION_NAME, properties.extensionName) == 0) {
                shader_float8_extension = ನಿಜ;
#ಆದರೆ_ಅಂತ್ಯ
            } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (strcmp("VK_VALVE_shader_mixed_float_dot_product", properties.extensionName) == 0 &&
                       !getenv("GGML_VK_DISABLE_DOT2")) {
                dot2_f16_support = ನಿಜ;
            } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (strcmp("VK_KHR_pipeline_executable_properties", properties.extensionName) == 0) {
                pipeline_executable_properties_support = ನಿಜ;
            } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (strcmp("VK_EXT_memory_priority", properties.extensionName) == 0 &&
                       getenv("GGML_VK_ENABLE_MEMORY_PRIORITY")) {
                device->ಸ್ಮರಣೆ_ಆದ್ಯತೆ = ನಿಜ;
            } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (strcmp("VK_EXT_external_memory_host", properties.extensionName) == 0) {
                device->external_ಸ್ಮರಣೆ_host = ನಿಜ;
#ಆದರೆ defined(VK_EXT_shader_64bit_indexing)
            } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (strcmp("VK_EXT_shader_64bit_indexing", properties.extensionName) == 0) {
                device->shader_64b_indexing = ನಿಜ;
#ಆದರೆ_ಅಂತ್ಯ
            } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (strcmp(VK_KHR_INTERNALLY_SYNCHRONIZED_QUEUES_EXTENSION_NAME, properties.extensionName) == 0) {
                internally_sync_support = ನಿಜ;
            } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (strcmp("VK_EXT_device_fault", properties.extensionName) == 0) {
                device->device_fault = ನಿಜ;
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
        vk::Physical_Device_Subgroup_ಗಾತ್ರ_Control_Properties_EXT subgroup_ಗಾತ್ರ_control_props;
        vk::Physical_Device_Shader_Integer_Dot_ಗುಣಲಬ್ಧ_Properties_KHR shader_integer_dot_ಗುಣಲಬ್ಧ_props;
        vk::Physical_Device_External_ಸ್ಮರಣೆ_Host_Properties_EXT external_ಸ್ಮರಣೆ_host_props;

        props2.p_ಮುಂದಿನ = &props3;
        props3.p_ಮುಂದಿನ = &subgroup_props;
        subgroup_props.p_ಮುಂದಿನ = &driver_props;
        driver_props.p_ಮುಂದಿನ = &vk11_props;
        vk11_props.p_ಮುಂದಿನ = &vk12_props;

        Vk_ತಳ_ಹೊರಹೋಗು_Structure * ಕೊನೆಯ_struct = (Vk_ತಳ_ಹೊರಹೋಗು_Structure *)&vk12_props;

        ಆದರೆ (maintenance4_support) {
            ಕೊನೆಯ_struct->p_ಮುಂದಿನ = (Vk_ತಳ_ಹೊರಹೋಗು_Structure *)&props4;
            ಕೊನೆಯ_struct = (Vk_ತಳ_ಹೊರಹೋಗು_Structure *)&props4;
        }
        ಆದರೆ (sm_builtins) {
            ಕೊನೆಯ_struct->p_ಮುಂದಿನ = (Vk_ತಳ_ಹೊರಹೋಗು_Structure *)&sm_props;
            ಕೊನೆಯ_struct = (Vk_ತಳ_ಹೊರಹೋಗು_Structure *)&sm_props;
        }
        ಆದರೆ (amd_shader_core_properties2) {
            ಕೊನೆಯ_struct->p_ಮುಂದಿನ = (Vk_ತಳ_ಹೊರಹೋಗು_Structure *)&amd_shader_core_properties2_props;
            ಕೊನೆಯ_struct = (Vk_ತಳ_ಹೊರಹೋಗು_Structure *)&amd_shader_core_properties2_props;
        }
        ಆದರೆ (device->subgroup_ಗಾತ್ರ_control) {
            ಕೊನೆಯ_struct->p_ಮುಂದಿನ = (Vk_ತಳ_ಹೊರಹೋಗು_Structure *)&subgroup_ಗಾತ್ರ_control_props;
            ಕೊನೆಯ_struct = (Vk_ತಳ_ಹೊರಹೋಗು_Structure *)&subgroup_ಗಾತ್ರ_control_props;
        }

#ಆದರೆ defined(VK_NV_cooperative_matrix2)
        vk::PhysicalDeviceCooperativeMatrix2PropertiesNV coopmat2_props;
        ಆದರೆ (coopmat2_support) {
            ಕೊನೆಯ_struct->p_ಮುಂದಿನ = (Vk_ತಳ_ಹೊರಹೋಗು_Structure *)&coopmat2_props;
            ಕೊನೆಯ_struct = (Vk_ತಳ_ಹೊರಹೋಗು_Structure *)&coopmat2_props;
        }
#ಆದರೆ_ಅಂತ್ಯ

        ಆದರೆ (device->integer_dot_ಗುಣಲಬ್ಧ) {
            ಕೊನೆಯ_struct->p_ಮುಂದಿನ = (Vk_ತಳ_ಹೊರಹೋಗು_Structure *)&shader_integer_dot_ಗುಣಲಬ್ಧ_props;
            ಕೊನೆಯ_struct = (Vk_ತಳ_ಹೊರಹೋಗು_Structure *)&shader_integer_dot_ಗುಣಲಬ್ಧ_props;
        }

        ಆದರೆ (device->external_ಸ್ಮರಣೆ_host) {
            ಕೊನೆಯ_struct->p_ಮುಂದಿನ = (Vk_ತಳ_ಹೊರಹೋಗು_Structure *)&external_ಸ್ಮರಣೆ_host_props;
            ಕೊನೆಯ_struct = (Vk_ತಳ_ಹೊರಹೋಗು_Structure *)&external_ಸ್ಮರಣೆ_host_props;
        }

        device->physical_device.getProperties2(&props2);
        device->properties = props2.properties;
        device->vendor_id = device->properties.vendorID;
        device->driver_id = driver_props.driverID;

        device->external_ಸ್ಮರಣೆ_host = device->external_ಸ್ಮರಣೆ_host && ggml_vk_external_ಸ್ಮರಣೆ_host_usable(device->driver_id);

        // Implementing the async backend interfaces seems broken on older Intel HW,
        // see https://github.com/ggml-org/llama.cpp/issues/17302.
        device->support_async = (device->vendor_id != VK_VENDOR_ID_INTEL ||
                                 std::string(device->properties.deviceName.ದತ್ತಾಂಶ()).find("(DG1)") == std::string::npos) &&
                                getenv("GGML_VK_DISABLE_ASYNC") == ಶೂನ್ಯಸೂಚಕ;

        ಆದರೆ (!device->support_async) {
            GGML_LOG_DEBUG("ggml_vulkan: WARNING: Async execution disabled on certain Intel devices.\n");
        }

        ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ* GGML_VK_FORCE_MAX_ಹಂಚಿಕೆ_ಗಾತ್ರ = getenv("GGML_VK_FORCE_MAX_ALLOCATION_SIZE");

        ಆದರೆ (GGML_VK_FORCE_MAX_ಹಂಚಿಕೆ_ಗಾತ್ರ != ಶೂನ್ಯಸೂಚಕ) {
            device->max_ಸ್ಮರಣೆ_ಹಂಚಿಕೆ_ಗಾತ್ರ = std::stoull(GGML_VK_FORCE_MAX_ಹಂಚಿಕೆ_ಗಾತ್ರ);
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (maintenance4_support) {
            device->max_ಸ್ಮರಣೆ_ಹಂಚಿಕೆ_ಗಾತ್ರ = std::min(props3.max_ಸ್ಮರಣೆ_ಹಂಚಿಕೆ_ಗಾತ್ರ_2, props4.max_ಬಫರ್_ಗಾತ್ರ_2);
        } ಇಲ್ಲದಿದ್ದರೆ {
            device->max_ಸ್ಮರಣೆ_ಹಂಚಿಕೆ_ಗಾತ್ರ = props3.max_ಸ್ಮರಣೆ_ಹಂಚಿಕೆ_ಗಾತ್ರ_2;
        }

        ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ* GGML_VK_FORCE_MAX_ಬಫರ್_ಗಾತ್ರ = getenv("GGML_VK_FORCE_MAX_BUFFER_SIZE");

        ಆದರೆ (GGML_VK_FORCE_MAX_ಬಫರ್_ಗಾತ್ರ != ಶೂನ್ಯಸೂಚಕ) {
            device->max_ಬಫರ್_ಗಾತ್ರ = std::stoull(GGML_VK_FORCE_MAX_ಬಫರ್_ಗಾತ್ರ);
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (maintenance4_support) {
            device->max_ಬಫರ್_ಗಾತ್ರ = props4.max_ಬಫರ್_ಗಾತ್ರ_2;
        } ಇಲ್ಲದಿದ್ದರೆ {
            device->max_ಬಫರ್_ಗಾತ್ರ = device->max_ಸ್ಮರಣೆ_ಹಂಚಿಕೆ_ಗಾತ್ರ;
        }

        ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ* GGML_VK_SUBALLOCATION_ಖಂಡ_ಗಾತ್ರ = getenv("GGML_VK_SUBALLOCATION_BLOCK_SIZE");

        ಆದರೆ (GGML_VK_SUBALLOCATION_ಖಂಡ_ಗಾತ್ರ != ಶೂನ್ಯಸೂಚಕ) {
            device->suballocation_ಖಂಡ_ಗಾತ್ರ = std::stoull(GGML_VK_SUBALLOCATION_ಖಂಡ_ಗಾತ್ರ);
        } ಇಲ್ಲದಿದ್ದರೆ {
            // Limit batching of allocations to 1GB by default to avoid fragmentation issues
            device->suballocation_ಖಂಡ_ಗಾತ್ರ = 1024*1024*1024;
        }
        device->suballocation_ಖಂಡ_ಗಾತ್ರ = std::min(device->suballocation_ಖಂಡ_ಗಾತ್ರ, device->max_ಸ್ಮರಣೆ_ಹಂಚಿಕೆ_ಗಾತ್ರ);

        device->subgroup_ಗಾತ್ರ = subgroup_props.subgroup_ಗಾತ್ರ_2;
        device->subgroup_ಗಾತ್ರ_log2 = uint32_t(log2f(ದಶಮಾಂಶ(device->subgroup_ಗಾತ್ರ)));
        device->uma = device->properties.deviceType == vk::PhysicalDeviceType::eIntegratedGpu;
        ಆದರೆ (sm_builtins) {
            device->shader_core_ಎಣಿಕೆ = sm_props.shaderSMCount;
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (amd_shader_core_properties2) {
            device->shader_core_ಎಣಿಕೆ = amd_shader_core_properties2_props.ಸಕ್ರಿಯ_Compute_Unit_ಎಣಿಕೆ;
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (device->vendor_id == VK_VENDOR_ID_INTEL) {
            device->shader_core_ಎಣಿಕೆ = ggml_vk_intel_shader_core_ಎಣಿಕೆ(device->physical_device);
        } ಇಲ್ಲದಿದ್ದರೆ {
            device->shader_core_ಎಣಿಕೆ = 0;
        }
        device->float_controls_rte_fp16 = vk12_props.shader_Rounding_ಬಹುಲಕ_RTEFloat16;
        device->float_controls_denorm_preserve_fp16 = vk12_props.shaderDenormPreserveFloat16;

        device->subgroup_basic = (vk11_props.subgroupSupportedStages & vk::Shader_Stage_Flag_ಬಿಟ್ಟುಗಳು::eCompute) &&
                                 (vk11_props.subgroupSupportedOperations & vk::Subgroup_Feature_Flag_ಬಿಟ್ಟುಗಳು::eBasic);
        device->subgroup_arithmetic = (vk11_props.subgroupSupportedStages & vk::Shader_Stage_Flag_ಬಿಟ್ಟುಗಳು::eCompute) &&
                                      (vk11_props.subgroupSupportedOperations & vk::Subgroup_Feature_Flag_ಬಿಟ್ಟುಗಳು::eArithmetic);
#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ __APPLE__
        // Workaround for subgroup arithmetic failing on MoltenVK with AMD GPUs (issue 15846)
        ಆದರೆ (device->vendor_id == VK_VENDOR_ID_AMD) {
            device->subgroup_arithmetic = ಸುಳ್ಳು;
        }
#ಆದರೆ_ಅಂತ್ಯ
        device->subgroup_shuffle = (vk11_props.subgroupSupportedStages & vk::Shader_Stage_Flag_ಬಿಟ್ಟುಗಳು::eCompute) &&
                                   (vk11_props.subgroupSupportedOperations & vk::Subgroup_Feature_Flag_ಬಿಟ್ಟುಗಳು::eShuffle);
#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ __APPLE__
        ಆದರೆ (device->vendor_id == VK_VENDOR_ID_AMD) {
            device->subgroup_shuffle = ಸುಳ್ಳು;
        }
#ಆದರೆ_ಅಂತ್ಯ
        device->subgroup_clustered = (vk11_props.subgroupSupportedStages & vk::Shader_Stage_Flag_ಬಿಟ್ಟುಗಳು::eCompute) &&
                                     (vk11_props.subgroupSupportedOperations & vk::Subgroup_Feature_Flag_ಬಿಟ್ಟುಗಳು::eClustered);

        device->subgroup_ballot = (vk11_props.subgroupSupportedStages & vk::Shader_Stage_Flag_ಬಿಟ್ಟುಗಳು::eCompute) &&
                                  (vk11_props.subgroupSupportedOperations & vk::Subgroup_Feature_Flag_ಬಿಟ್ಟುಗಳು::eBallot);

        device->subgroup_vote = (vk11_props.subgroupSupportedStages & vk::Shader_Stage_Flag_ಬಿಟ್ಟುಗಳು::eCompute) &&
                                (vk11_props.subgroupSupportedOperations & vk::Subgroup_Feature_Flag_ಬಿಟ್ಟುಗಳು::eVote);

        // Submit at least every 100 nodes, in case there are workloads without as much matmul.
        device->max_nodes_ಪ್ರತಿ_submit = 100;
        ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ* GGML_VK_MAX_NODES_ಪ್ರತಿ_SUBMIT = getenv("GGML_VK_MAX_NODES_PER_SUBMIT");
        ಆದರೆ (GGML_VK_MAX_NODES_ಪ್ರತಿ_SUBMIT != ಶೂನ್ಯಸೂಚಕ) {
            uint32_t max_nodes_ಪ್ರತಿ_submit = std::stoul(GGML_VK_MAX_NODES_ಪ್ರತಿ_SUBMIT);
            device->max_nodes_ಪ್ರತಿ_submit = std::max(max_nodes_ಪ್ರತಿ_submit, 1u);
        }

        ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ force_disable_f16 = getenv("GGML_VK_DISABLE_F16") != ಶೂನ್ಯಸೂಚಕ;

        device->fp16 = !force_disable_f16 && fp16_storage && fp16_compute;

        ಆದರೆ (!ggml_vk_khr_cooperative_ಮಾತೃಕೆ_support(device->properties, driver_props, device->architecture)) {
            device->coopmat_support = ಸುಳ್ಳು;
        }

        device->integer_dot_ಗುಣಲಬ್ಧ = device->integer_dot_ಗುಣಲಬ್ಧ && shader_integer_dot_ಗುಣಲಬ್ಧ_props.integer_Dot_Product4x8_ಬಿಟ್_Packed_Signed_Accelerated;

        device->min_imported_host_ಸೂಚಕ_alignment = external_ಸ್ಮರಣೆ_host_props.min_Imported_Host_ಸೂಚಕ_Alignment;

        device->max_workgroup_ಗಾತ್ರ_log2 = uint32_t(log2f(ದಶಮಾಂಶ(device->properties.limits.max_Compute_Work_ಗುಂಪು_Invocations)));

        std::vector<vk::ಸರತಿ_Family_Properties> ಸರತಿ_family_props = device->physical_device.get_ಸರತಿ_Family_Properties();

        // Try to find a non-graphics compute queue and transfer-focused queues
        // Allow overriding avoiding the graphics queue because it can increase performance on RADV
        ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ allow_graphics_ಸರತಿ = (getenv("GGML_VK_ALLOW_GRAPHICS_QUEUE") != ಶೂನ್ಯಸೂಚಕ);
        ಸ್ಥಿರಾಂಕ vk::ಸರತಿ_Flag_ಬಿಟ್ಟುಗಳು graphics_flag = allow_graphics_ಸರತಿ ? (vk::ಸರತಿ_Flag_ಬಿಟ್ಟುಗಳು)0 : vk::ಸರತಿ_Flag_ಬಿಟ್ಟುಗಳು::eGraphics;
        ಸ್ಥಿರಾಂಕ uint32_t compute_ಸರತಿ_family_ಸೂಚ್ಯಂಕ = ggml_vk_ಕಂಡುಹಿಡಿ_ಸರತಿ_family_ಸೂಚ್ಯಂಕ(ಸರತಿ_family_props, vk::ಸರತಿ_Flag_ಬಿಟ್ಟುಗಳು::eCompute, graphics_flag, -1, 1);
        ಸ್ಥಿರಾಂಕ uint32_t transfer_ಸರತಿ_family_ಸೂಚ್ಯಂಕ = ggml_vk_ಕಂಡುಹಿಡಿ_ಸರತಿ_family_ಸೂಚ್ಯಂಕ(ಸರತಿ_family_props, vk::ಸರತಿ_Flag_ಬಿಟ್ಟುಗಳು::eTransfer, vk::ಸರತಿ_Flag_ಬಿಟ್ಟುಗಳು::eCompute | graphics_flag, compute_ಸರತಿ_family_ಸೂಚ್ಯಂಕ, 1);

        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ priorities[] = { 1.0f, 1.0f };
        device->single_ಸರತಿ = compute_ಸರತಿ_family_ಸೂಚ್ಯಂಕ == transfer_ಸರತಿ_family_ಸೂಚ್ಯಂಕ && ಸರತಿ_family_props[compute_ಸರತಿ_family_ಸೂಚ್ಯಂಕ].ಸರತಿ_ಎಣಿಕೆ == 1;

        std::vector<vk::Device_ಸರತಿ_Create_Info> device_ಸರತಿ_create_infos;
        vk::DeviceCreateInfo device_create_info{};
        std::vector<ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *> device_extensions;
        vk::PhysicalDeviceFeatures device_features = device->physical_device.getFeatures();

        VkPhysicalDeviceFeatures2 device_features2;
        device_features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
        device_features2.p_ಮುಂದಿನ = ಶೂನ್ಯಸೂಚಕ;
        device_features2.features = (VkPhysicalDeviceFeatures)device_features;

        VkPhysicalDeviceVulkan11Features vk11_features;
        vk11_features.p_ಮುಂದಿನ = ಶೂನ್ಯಸೂಚಕ;
        vk11_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
        device_features2.p_ಮುಂದಿನ = &vk11_features;

        VkPhysicalDeviceVulkan12Features vk12_features;
        vk12_features.p_ಮುಂದಿನ = ಶೂನ್ಯಸೂಚಕ;
        vk12_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
        vk11_features.p_ಮುಂದಿನ = &vk12_features;

        ಕೊನೆಯ_struct = (Vk_ತಳ_ಹೊರಹೋಗು_Structure *)&vk12_features;

        VkPhysicalDeviceInternallySynchronizedQueuesFeaturesKHR internally_synchronized_queues_features{};
        internally_synchronized_queues_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_INTERNALLY_SYNCHRONIZED_QUEUES_FEATURES_KHR;
        internally_synchronized_queues_features.p_ಮುಂದಿನ = ಶೂನ್ಯಸೂಚಕ;
        internally_synchronized_queues_features.internallySynchronizedQueues = VK_FALSE;

        ಆದರೆ (internally_sync_support) {
            ಕೊನೆಯ_struct->p_ಮುಂದಿನ = (Vk_ತಳ_ಹೊರಹೋಗು_Structure *)&internally_synchronized_queues_features;
            ಕೊನೆಯ_struct = (Vk_ತಳ_ಹೊರಹೋಗು_Structure *)&internally_synchronized_queues_features;
            device_extensions.push_back(VK_KHR_INTERNALLY_SYNCHRONIZED_QUEUES_EXTENSION_NAME);
        }

        VkPhysicalDevicePipelineRobustnessFeaturesEXT pl_robustness_features;
        pl_robustness_features.p_ಮುಂದಿನ = ಶೂನ್ಯಸೂಚಕ;
        pl_robustness_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_ROBUSTNESS_FEATURES_EXT;
        pl_robustness_features.pipelineRobustness = VK_FALSE;

        ಆದರೆ (pipeline_robustness) {
            ಕೊನೆಯ_struct->p_ಮುಂದಿನ = (Vk_ತಳ_ಹೊರಹೋಗು_Structure *)&pl_robustness_features;
            ಕೊನೆಯ_struct = (Vk_ತಳ_ಹೊರಹೋಗು_Structure *)&pl_robustness_features;
            device_extensions.push_back("VK_EXT_pipeline_robustness");
        }

        Vk_Physical_Device_ಸ್ಮರಣೆ_ಆದ್ಯತೆ_Features_EXT ಸ್ಮರಣೆ_ಆದ್ಯತೆ_features;
        ಸ್ಮರಣೆ_ಆದ್ಯತೆ_features.p_ಮುಂದಿನ = ಶೂನ್ಯಸೂಚಕ;
        ಸ್ಮರಣೆ_ಆದ್ಯತೆ_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ಸ್ಮರಣೆ_ಆದ್ಯತೆ_FEATURES_EXT;
        ಸ್ಮರಣೆ_ಆದ್ಯತೆ_features.ಸ್ಮರಣೆ_ಆದ್ಯತೆ_2 = VK_FALSE;
        ಆದರೆ (device->ಸ್ಮರಣೆ_ಆದ್ಯತೆ) {
            ಕೊನೆಯ_struct->p_ಮುಂದಿನ = (Vk_ತಳ_ಹೊರಹೋಗು_Structure *)&ಸ್ಮರಣೆ_ಆದ್ಯತೆ_features;
            ಕೊನೆಯ_struct = (Vk_ತಳ_ಹೊರಹೋಗು_Structure *)&ಸ್ಮರಣೆ_ಆದ್ಯತೆ_features;
            device_extensions.push_back("VK_EXT_memory_priority");
        }

        Vk_Physical_Device_Subgroup_ಗಾತ್ರ_Control_Features_EXT subgroup_ಗಾತ್ರ_control_features;
        subgroup_ಗಾತ್ರ_control_features.p_ಮುಂದಿನ = ಶೂನ್ಯಸೂಚಕ;
        subgroup_ಗಾತ್ರ_control_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_ಗಾತ್ರ_CONTROL_FEATURES_EXT;
        subgroup_ಗಾತ್ರ_control_features.compute_ತುಂಬಿದ_Subgroups = ಸುಳ್ಳು;
        subgroup_ಗಾತ್ರ_control_features.subgroup_ಗಾತ್ರ_Control = ಸುಳ್ಳು;

        ಆದರೆ (device->subgroup_ಗಾತ್ರ_control) {
            ಕೊನೆಯ_struct->p_ಮುಂದಿನ = (Vk_ತಳ_ಹೊರಹೋಗು_Structure *)&subgroup_ಗಾತ್ರ_control_features;
            ಕೊನೆಯ_struct = (Vk_ತಳ_ಹೊರಹೋಗು_Structure *)&subgroup_ಗಾತ್ರ_control_features;
        }

#ಆದರೆ defined(VK_KHR_cooperative_ಮಾತೃಕೆ)
        Vk_Physical_Device_Cooperative_ಮಾತೃಕೆ_Features_KHR coopmat_features;
        coopmat_features.p_ಮುಂದಿನ = ಶೂನ್ಯಸೂಚಕ;
        coopmat_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_ಮಾತೃಕೆ_FEATURES_KHR;
        coopmat_features.cooperative_ಮಾತೃಕೆ_2 = VK_FALSE;

        ಆದರೆ (device->coopmat_support) {
            ಕೊನೆಯ_struct->p_ಮುಂದಿನ = (Vk_ತಳ_ಹೊರಹೋಗು_Structure *)&coopmat_features;
            ಕೊನೆಯ_struct = (Vk_ತಳ_ಹೊರಹೋಗು_Structure *)&coopmat_features;
        }
#ಆದರೆ_ಅಂತ್ಯ

#ಆದರೆ defined(VK_NV_cooperative_matrix2)
        VkPhysicalDeviceCooperativeMatrix2FeaturesNV coopmat2_features {};
        coopmat2_features.p_ಮುಂದಿನ = ಶೂನ್ಯಸೂಚಕ;
        coopmat2_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_ಮಾತೃಕೆ_2_FEATURES_NV;
        ಆದರೆ (coopmat2_support) {
            ಕೊನೆಯ_struct->p_ಮುಂದಿನ = (Vk_ತಳ_ಹೊರಹೋಗು_Structure *)&coopmat2_features;
            ಕೊನೆಯ_struct = (Vk_ತಳ_ಹೊರಹೋಗು_Structure *)&coopmat2_features;
            device_extensions.push_back("VK_NV_cooperative_matrix2");
        }
#ಆದರೆ_ಅಂತ್ಯ

        Vk_Physical_Device_Cooperative_ಮಾತೃಕೆ_Decode_ಸದಿಶ_Features_NV coopmat2_decode_ಸದಿಶ_features {};
        coopmat2_decode_ಸದಿಶ_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_ಮಾತೃಕೆ_DECODE_ಸದಿಶ_FEATURES_NV;
        ಆದರೆ (coopmat2_decode_ಸದಿಶ_support) {
            ಕೊನೆಯ_struct->p_ಮುಂದಿನ = (Vk_ತಳ_ಹೊರಹೋಗು_Structure *)&coopmat2_decode_ಸದಿಶ_features;
            ಕೊನೆಯ_struct = (Vk_ತಳ_ಹೊರಹೋಗು_Structure *)&coopmat2_decode_ಸದಿಶ_features;
            device_extensions.push_back(VK_NV_COOPERATIVE_ಮಾತೃಕೆ_DECODE_ಸದಿಶ_EXTENSION_NAME);
        }

#ಆದರೆ defined(VK_KHR_shader_bfloat16)
        VkPhysicalDeviceShaderBfloat16FeaturesKHR bfloat16_features {};
        bfloat16_features.p_ಮುಂದಿನ = ಶೂನ್ಯಸೂಚಕ;
        bfloat16_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_BFLOAT16_FEATURES_KHR;
        ಆದರೆ (bfloat16_support) {
            ಕೊನೆಯ_struct->p_ಮುಂದಿನ = (Vk_ತಳ_ಹೊರಹೋಗು_Structure *)&bfloat16_features;
            ಕೊನೆಯ_struct = (Vk_ತಳ_ಹೊರಹೋಗು_Structure *)&bfloat16_features;
            device_extensions.push_back("VK_KHR_shader_bfloat16");
        }
#ಆದರೆ_ಅಂತ್ಯ

        VkPhysicalDeviceShaderOCPMicroscalingTypesFeaturesEXT ocp_microscaling_features {};
        ocp_microscaling_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_OCP_MICROSCALING_TYPES_FEATURES_EXT;
        ಆದರೆ (ocp_microscaling_extension) {
            ಕೊನೆಯ_struct->p_ಮುಂದಿನ = (Vk_ತಳ_ಹೊರಹೋಗು_Structure *)&ocp_microscaling_features;
            ಕೊನೆಯ_struct = (Vk_ತಳ_ಹೊರಹೋಗು_Structure *)&ocp_microscaling_features;
            device_extensions.push_back(VK_EXT_SHADER_OCP_MICROSCALING_TYPES_EXTENSION_NAME);
        }

        VkPhysicalDeviceShaderFloat8FeaturesEXT shader_float8_features {};
        shader_float8_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT8_FEATURES_EXT;
        ಆದರೆ (shader_float8_extension) {
            ಕೊನೆಯ_struct->p_ಮುಂದಿನ = (Vk_ತಳ_ಹೊರಹೋಗು_Structure *)&shader_float8_features;
            ಕೊನೆಯ_struct = (Vk_ತಳ_ಹೊರಹೋಗು_Structure *)&shader_float8_features;
            device_extensions.push_back(VK_EXT_SHADER_FLOAT8_EXTENSION_NAME);
        }

        VkPhysicalDeviceMaintenance4Features maint4_features {};
        maint4_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_4_FEATURES;
        ಆದರೆ (maintenance4_support) {
            ಕೊನೆಯ_struct->p_ಮುಂದಿನ = (Vk_ತಳ_ಹೊರಹೋಗು_Structure *)&maint4_features;
            ಕೊನೆಯ_struct = (Vk_ತಳ_ಹೊರಹೋಗು_Structure *)&maint4_features;
            device_extensions.push_back("VK_KHR_maintenance4");
        }

        Vk_Physical_Device_Shader_Integer_Dot_ಗುಣಲಬ್ಧ_Features_KHR shader_integer_dot_ಗುಣಲಬ್ಧ_features {};
        shader_integer_dot_ಗುಣಲಬ್ಧ_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_ಗುಣಲಬ್ಧ_FEATURES_KHR;
        ಆದರೆ (device->integer_dot_ಗುಣಲಬ್ಧ) {
            ಕೊನೆಯ_struct->p_ಮುಂದಿನ = (Vk_ತಳ_ಹೊರಹೋಗು_Structure *)&shader_integer_dot_ಗುಣಲಬ್ಧ_features;
            ಕೊನೆಯ_struct = (Vk_ತಳ_ಹೊರಹೋಗು_Structure *)&shader_integer_dot_ಗುಣಲಬ್ಧ_features;
            device_extensions.push_back("VK_KHR_shader_integer_dot_product");
        }

        Vk_Physical_Device_Shader_Mixed_Float_Dot_ಗುಣಲಬ್ಧ_Features_VALVE dot2_features {};
        dot2_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_MIXED_FLOAT_DOT_ಗುಣಲಬ್ಧ_FEATURES_VALVE;
        ಆದರೆ (dot2_f16_support) {
            ಕೊನೆಯ_struct->p_ಮುಂದಿನ = (Vk_ತಳ_ಹೊರಹೋಗು_Structure *)&dot2_features;
            ಕೊನೆಯ_struct = (Vk_ತಳ_ಹೊರಹೋಗು_Structure *)&dot2_features;
            device_extensions.push_back("VK_VALVE_shader_mixed_float_dot_product");
        }

        VkPhysicalDevicePipelineExecutablePropertiesFeaturesKHR pep_features {};
        pep_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_EXECUTABLE_PROPERTIES_FEATURES_KHR;
        ಆದರೆ (pipeline_executable_properties_support) {
            ಕೊನೆಯ_struct->p_ಮುಂದಿನ = (Vk_ತಳ_ಹೊರಹೋಗು_Structure *)&pep_features;
            ಕೊನೆಯ_struct = (Vk_ತಳ_ಹೊರಹೋಗು_Structure *)&pep_features;
            device_extensions.push_back("VK_KHR_pipeline_executable_properties");
        }

        ಆದರೆ (device->external_ಸ್ಮರಣೆ_host) {
            device_extensions.push_back("VK_EXT_external_memory_host");
        }

#ಆದರೆ defined(VK_EXT_shader_64bit_indexing)
        Vk_Physical_Device_Shader64_ಬಿಟ್_Indexing_Features_EXT shader_64bit_indexing_features {};
        shader_64bit_indexing_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_64_ಬಿಟ್_INDEXING_FEATURES_EXT;
        ಆದರೆ (device->shader_64b_indexing) {
            ಕೊನೆಯ_struct->p_ಮುಂದಿನ = (Vk_ತಳ_ಹೊರಹೋಗು_Structure *)&shader_64bit_indexing_features;
            ಕೊನೆಯ_struct = (Vk_ತಳ_ಹೊರಹೋಗು_Structure *)&shader_64bit_indexing_features;
            device_extensions.push_back("VK_EXT_shader_64bit_indexing");
        }
#ಆದರೆ_ಅಂತ್ಯ

        VkPhysicalDeviceFaultFeaturesEXT fault_features {};
        fault_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FAULT_FEATURES_EXT;
        ಆದರೆ (device->device_fault) {
            ಕೊನೆಯ_struct->p_ಮುಂದಿನ = (Vk_ತಳ_ಹೊರಹೋಗು_Structure *)&fault_features;
            ಕೊನೆಯ_struct = (Vk_ತಳ_ಹೊರಹೋಗು_Structure *)&fault_features;
            device_extensions.push_back("VK_EXT_device_fault");
        }

        vkGetPhysicalDeviceFeatures2(device->physical_device, &device_features2);

        device->device_fault = device->device_fault && fault_features.deviceFault;

        device->has_internally_synchronized_queues = internally_synchronized_queues_features.internallySynchronizedQueues;

        // Build queue create infos only after querying whether internally synchronized queues are enabled.
        // getQueue2() later uses the same flag, so creation/retrieval must stay consistent.
        vk::Device_ಸರತಿ_Create_Flags ಸರತಿ_flags = device->has_internally_synchronized_queues ?
                                                eInternallySynchronizedKHR :
                                                vk::Device_ಸರತಿ_Create_Flags();

        ಆದರೆ (compute_ಸರತಿ_family_ಸೂಚ್ಯಂಕ != transfer_ಸರತಿ_family_ಸೂಚ್ಯಂಕ) {
            device_ಸರತಿ_create_infos.push_back({ಸರತಿ_flags, compute_ಸರತಿ_family_ಸೂಚ್ಯಂಕ, 1, priorities});
            device_ಸರತಿ_create_infos.push_back({ಸರತಿ_flags, transfer_ಸರತಿ_family_ಸೂಚ್ಯಂಕ, 1, priorities + 1});
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ(!device->single_ಸರತಿ) {
            device_ಸರತಿ_create_infos.push_back({ಸರತಿ_flags, compute_ಸರತಿ_family_ಸೂಚ್ಯಂಕ, 2, priorities});
        } ಇಲ್ಲದಿದ್ದರೆ {
            device_ಸರತಿ_create_infos.push_back({ಸರತಿ_flags, compute_ಸರತಿ_family_ಸೂಚ್ಯಂಕ, 1, priorities});
        }

        device->pipeline_executable_properties_support = pipeline_executable_properties_support;

        device->fp16 = device->fp16 && vk12_features.shaderFloat16;

#ಆದರೆ defined(VK_KHR_shader_bfloat16)
        device->bf16 = bfloat16_support && bfloat16_features.shaderBFloat16Type;
#ಇಲ್ಲದಿದ್ದರೆ
        device->bf16 = ಸುಳ್ಳು;
#ಆದರೆ_ಅಂತ್ಯ

        device->dot2_f16 = dot2_f16_support && dot2_features.shader_Mixed_Float_Dot_ಗುಣಲಬ್ಧ_Float16_Acc_Float32;
        device->ocp_fp4 = ocp_microscaling_extension && ocp_microscaling_features.shaderFloat4 &&
                          shader_float8_extension && shader_float8_features.shaderFloat8 &&
                          !getenv("GGML_VK_DISABLE_OCP_FP4");

        device->pipeline_robustness = pl_robustness_features.pipelineRobustness;

        device->multi_ಕೂಡಿಸು = vk12_props.shader_Rounding_ಬಹುಲಕ_RTEFloat16 &&
                            device->properties.limits.max_ತಳ್ಳು_Constants_ಗಾತ್ರ >= ಗಾತ್ರ(vk_op_multi_ಕೂಡಿಸು_ತಳ್ಳು_constants) &&
                            getenv("GGML_VK_DISABLE_MULTI_ADD") == ಶೂನ್ಯಸೂಚಕ;

        device->shader_int64 = device_features2.features.shaderInt64;
        device->ಬಫರ್_device_ವಿಳಾಸ = vk12_features.ಬಫರ್_Device_ವಿಳಾಸ;
        device->vulkan_ಸ್ಮರಣೆ_model = vk12_features.vulkan_ಸ್ಮರಣೆ_Model;

        ಆದರೆ (device->subgroup_ಗಾತ್ರ_control) {
            device->subgroup_min_ಗಾತ್ರ = subgroup_ಗಾತ್ರ_control_props.min_Subgroup_ಗಾತ್ರ;
            device->subgroup_max_ಗಾತ್ರ = subgroup_ಗಾತ್ರ_control_props.max_Subgroup_ಗಾತ್ರ;
            device_extensions.push_back("VK_EXT_subgroup_size_control");
        }

        device->subgroup_ಗಾತ್ರ_control = device->subgroup_ಗಾತ್ರ_control &&
                (subgroup_ಗಾತ್ರ_control_props.required_Subgroup_ಗಾತ್ರ_Stages & vk::Shader_Stage_Flag_ಬಿಟ್ಟುಗಳು::eCompute) &&
                subgroup_ಗಾತ್ರ_control_features.subgroup_ಗಾತ್ರ_Control;

        device->subgroup_require_ತುಂಬಿದ_support = subgroup_ಗಾತ್ರ_control_features.compute_ತುಂಬಿದ_Subgroups;

#ಆದರೆ defined(VK_KHR_cooperative_ಮಾತೃಕೆ)
        device->coopmat_support = device->coopmat_support && coopmat_features.cooperative_ಮಾತೃಕೆ_2;
        device->coopmat1_fa_support = device->coopmat_support && device->subgroup_require_ತುಂಬಿದ_support;
#ಆದರೆ_ಅಂತ್ಯ

        ಆದರೆ (coopmat2_support) {
#ಆದರೆ defined(VK_NV_cooperative_matrix2) && defined(GGML_VULKAN_COOPMAT2_GLSLC_SUPPORT)
            ಆದರೆ (coopmat2_features.cooperative_ಮಾತೃಕೆ_Workgroup_Scope &&
                coopmat2_features.cooperative_ಮಾತೃಕೆ_Flexible_Dimensions &&
                coopmat2_features.cooperative_ಮಾತೃಕೆ_Reductions &&
                coopmat2_features.cooperative_ಮಾತೃಕೆ_Conversions &&
                coopmat2_features.cooperative_ಮಾತೃಕೆ_ಪ್ರತಿ_Element_Operations &&
                coopmat2_features.cooperative_ಮಾತೃಕೆ_Tensor_Addressing &&
                coopmat2_features.cooperative_ಮಾತೃಕೆ_ಖಂಡ_Loads &&
                vk12_features.ಬಫರ್_Device_ವಿಳಾಸ) {

                std::vector<Vk_Cooperative_ಮಾತೃಕೆ_Flexible_Dimensions_Properties_NV> flexible_dimensions;
                uint32_t count = 0;

                PFN_vk_Get_Physical_Device_Cooperative_ಮಾತೃಕೆ_Flexible_Dimensions_Properties_NV
                    vk_Get_Physical_Device_Cooperative_ಮಾತೃಕೆ_Flexible_Dimensions_Properties_NV =
                        (PFN_vk_Get_Physical_Device_Cooperative_ಮಾತೃಕೆ_Flexible_Dimensions_Properties_NV)
                        vk_instance.instance.getProcAddr("vkGetPhysicalDeviceCooperativeMatrixFlexibleDimensionsPropertiesNV");

                vk_Get_Physical_Device_Cooperative_ಮಾತೃಕೆ_Flexible_Dimensions_Properties_NV(device->physical_device, &count, ಶೂನ್ಯಸೂಚಕ);

                Vk_Cooperative_ಮಾತೃಕೆ_Flexible_Dimensions_Properties_NV ಖಾಲಿಗಳು_prop {};
                ಖಾಲಿಗಳು_prop.sType = VK_STRUCTURE_TYPE_COOPERATIVE_ಮಾತೃಕೆ_FLEXIBLE_DIMENSIONS_PROPERTIES_NV;
                flexible_dimensions.resize(count, ಖಾಲಿಗಳು_prop);

                vk_Get_Physical_Device_Cooperative_ಮಾತೃಕೆ_Flexible_Dimensions_Properties_NV(device->physical_device, &count, flexible_dimensions.ದತ್ತಾಂಶ());

                ತಾರ್ಕಿಕ ಸಿಕ್ಕಿತು_fp16_128 = ಸುಳ್ಳು,
                     ಸಿಕ್ಕಿತು_fp16_256 = ಸುಳ್ಳು,
                     ಸಿಕ್ಕಿತು_fp32_128 = ಸುಳ್ಳು,
                     ಸಿಕ್ಕಿತು_fp32_256 = ಸುಳ್ಳು;
                ತಾರ್ಕಿಕ ಸಿಕ್ಕಿತು_bf16_128 = ಸುಳ್ಳು,
                     ಸಿಕ್ಕಿತು_bf16_256 = ಸುಳ್ಳು;
                // need to support fp16*fp16 with fp16/fp32 accumulator, for workgroupsize 128
                // with 32x16x16 and 256 with 32x32x16.
                ಗಾಗಿ (ಸ್ವಯಂಚಾಲಿತ &prop : flexible_dimensions) {
                    ಆದರೆ (prop.saturatingAccumulation == VK_FALSE &&
                        prop.scope == VK_SCOPE_WORKGROUP_KHR) {

                        ಆದರೆ (prop.AType == VK_COMPONENT_TYPE_FLOAT16_KHR &&
                            prop.BType == VK_COMPONENT_TYPE_FLOAT16_KHR) {

                            ಆದರೆ (prop.workgroupInvocations == 128 &&
                                prop.MGranularity <= 32 &&
                                prop.NGranularity <= 16 &&
                                prop.KGranularity <= 16) {
                                ಆದರೆ (prop.CType == VK_COMPONENT_TYPE_FLOAT16_KHR &&
                                    prop.ಫಲಿತಾಂಶ_Type == VK_COMPONENT_TYPE_FLOAT16_KHR) {
                                    ಸಿಕ್ಕಿತು_fp16_128 = ನಿಜ;
                                }
                                ಆದರೆ (prop.CType == VK_COMPONENT_TYPE_FLOAT32_KHR &&
                                    prop.ಫಲಿತಾಂಶ_Type == VK_COMPONENT_TYPE_FLOAT32_KHR) {
                                    ಸಿಕ್ಕಿತು_fp32_128 = ನಿಜ;
                                }
                            }
                            ಆದರೆ (prop.workgroupInvocations == 256 &&
                                prop.MGranularity <= 32 &&
                                prop.NGranularity <= 32 &&
                                prop.KGranularity <= 16) {
                                ಆದರೆ (prop.CType == VK_COMPONENT_TYPE_FLOAT16_KHR &&
                                    prop.ಫಲಿತಾಂಶ_Type == VK_COMPONENT_TYPE_FLOAT16_KHR) {
                                    ಸಿಕ್ಕಿತು_fp16_256 = ನಿಜ;
                                }
                                ಆದರೆ (prop.CType == VK_COMPONENT_TYPE_FLOAT32_KHR &&
                                    prop.ಫಲಿತಾಂಶ_Type == VK_COMPONENT_TYPE_FLOAT32_KHR) {
                                    ಸಿಕ್ಕಿತು_fp32_256 = ನಿಜ;
                                }
                            }
                        }

#ಆದರೆ defined(VK_KHR_shader_bfloat16) && defined(GGML_VULKAN_BFLOAT16_GLSLC_SUPPORT)
                        ಆದರೆ (bfloat16_support &&
                            prop.AType == VK_COMPONENT_TYPE_BFLOAT16_KHR &&
                            prop.BType == VK_COMPONENT_TYPE_BFLOAT16_KHR &&
                            prop.CType == VK_COMPONENT_TYPE_FLOAT32_KHR &&
                            prop.ಫಲಿತಾಂಶ_Type == VK_COMPONENT_TYPE_FLOAT32_KHR) {

                            ಆದರೆ (prop.workgroupInvocations == 128 &&
                                prop.MGranularity <= 32 &&
                                prop.NGranularity <= 16 &&
                                prop.KGranularity <= 16) {
                                ಸಿಕ್ಕಿತು_bf16_128 = ನಿಜ;
                            }
                            ಆದರೆ (prop.workgroupInvocations == 256 &&
                                prop.MGranularity <= 32 &&
                                prop.NGranularity <= 32 &&
                                prop.KGranularity <= 16) {
                                ಸಿಕ್ಕಿತು_bf16_256 = ನಿಜ;
                            }
                        }
#ಆದರೆ_ಅಂತ್ಯ
                    }
                }
                ಆದರೆ (ಸಿಕ್ಕಿತು_fp16_128 && ಸಿಕ್ಕಿತು_fp16_256 &&
                    ಸಿಕ್ಕಿತು_fp32_128 && ಸಿಕ್ಕಿತು_fp32_256 &&
                    coopmat2_props.cooperative_ಮಾತೃಕೆ_Flexible_Dimensions_Max_Dimension >= 512) {
                    device->coopmat2 = ನಿಜ;
                    device->coopmat2_bf16_support = ಸಿಕ್ಕಿತು_bf16_128 && ಸಿಕ್ಕಿತು_bf16_256;
                    device->coopmat2_decode_ಸದಿಶ = coopmat2_decode_ಸದಿಶ_support && coopmat2_decode_ಸದಿಶ_features.cooperative_ಮಾತೃಕೆ_Decode_ಸದಿಶ;
                }
            }
#ಆದರೆ_ಅಂತ್ಯ
        }

        ಆದರೆ (!vk11_features.storage_Buffer16_ಬಿಟ್_Access) {
            std::cerr << "ggml_vulkan: device " << GGML_VK_NAME << idx << " does not support 16-bit storage." << std::ಸಾಲು_ಅಂತ್ಯ;
            ಎಸೆ std::runtime_error("Unsupported device");
        }

        device_extensions.push_back("VK_KHR_16bit_storage");

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_VULKAN_VALIDATE
        device_extensions.push_back("VK_KHR_shader_non_semantic_info");
#ಆದರೆ_ಅಂತ್ಯ

        ಆದರೆ (device->fp16) {
            device_extensions.push_back("VK_KHR_shader_float16_int8");
        }

#ಆದರೆ defined(VK_KHR_cooperative_ಮಾತೃಕೆ)
        ಆದರೆ (device->coopmat_support) {
            // Query supported shapes
            std::vector<Vk_Cooperative_ಮಾತೃಕೆ_Properties_KHR> cm_props;

            PFN_vk_Get_Physical_Device_Cooperative_ಮಾತೃಕೆ_Properties_KHR pfn_vk_Get_Physical_Device_Cooperative_ಮಾತೃಕೆ_Properties_KHR =
                (PFN_vk_Get_Physical_Device_Cooperative_ಮಾತೃಕೆ_Properties_KHR)vkGetInstanceProcAddr(vk_instance.instance, "vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR");

            uint32_t cm_props_ಅಂಕೆ;

            pfn_vk_Get_Physical_Device_Cooperative_ಮಾತೃಕೆ_Properties_KHR(device->physical_device, &cm_props_ಅಂಕೆ, ಶೂನ್ಯಸೂಚಕ);

            cm_props.resize(cm_props_ಅಂಕೆ);

            ಗಾಗಿ (ಸ್ವಯಂಚಾಲಿತ& prop : cm_props) {
                prop.sType = VK_STRUCTURE_TYPE_COOPERATIVE_ಮಾತೃಕೆ_PROPERTIES_KHR;
            }

            pfn_vk_Get_Physical_Device_Cooperative_ಮಾತೃಕೆ_Properties_KHR(device->physical_device, &cm_props_ಅಂಕೆ, cm_props.ದತ್ತಾಂಶ());

            VK_LOG_DEBUG("ggml_vulkan: Cooperative Matrix Shapes: " << cm_props.size());

            ಗಾಗಿ (ಸ್ವಯಂಚಾಲಿತ& prop : cm_props) {
                VK_LOG_DEBUG("ggml_vulkan: M: " << prop.MSize << " N: " << prop.NSize << " K: " << prop.KSize << " A: " << vk::to_string((vk::ComponentTypeKHR)prop.AType) << " B: " << vk::to_string((vk::ComponentTypeKHR)prop.BType) << " C: " << vk::to_string((vk::ComponentTypeKHR)prop.CType) << " Result: " << vk::to_string((vk::ComponentTypeKHR)prop.ಫಲಿತಾಂಶ_Type) << " saturatingAccumulation: " << prop.saturatingAccumulation << " scope: " << vk::to_string((vk::ScopeKHR)prop.scope));

                ಆದರೆ ((vk::ComponentTypeKHR)prop.AType == vk::ComponentTypeKHR::eFloat16 &&
                    (vk::ComponentTypeKHR)prop.BType == vk::ComponentTypeKHR::eFloat16 &&
                    (vk::ScopeKHR)prop.scope == vk::ScopeKHR::eSubgroup
                ) {
                    ಆದರೆ ((vk::ComponentTypeKHR)prop.CType == vk::ComponentTypeKHR::eFloat32 &&
                        (vk::ComponentTypeKHR)prop.ಫಲಿತಾಂಶ_Type == vk::ComponentTypeKHR::eFloat32) {
                        // coopmat sizes not set yet
                        ಆದರೆ (device->coopmat_m == 0) {
                            device->coopmat_acc_f32_support = ನಿಜ;
                            device->coopmat_m = prop.MSize;
                            device->coopmat_n = prop.NSize;
                            device->coopmat_k = prop.KSize;
                        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (device->coopmat_m == prop.MSize && device->coopmat_n == prop.NSize && device->coopmat_k == prop.KSize) {
                            // Only enable if shape is identical
                            device->coopmat_acc_f32_support = ನಿಜ;
                        }
                        ಆದರೆ (prop.MSize == 16 && prop.NSize == 16 && prop.KSize == 16) {
                            device->coopmat_support_16x16x16_f32acc = ನಿಜ;
                        }
                    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ ((vk::ComponentTypeKHR)prop.CType == vk::ComponentTypeKHR::eFloat16 &&
                               (vk::ComponentTypeKHR)prop.ಫಲಿತಾಂಶ_Type == vk::ComponentTypeKHR::eFloat16) {
                        // coopmat sizes not set yet
                        ಆದರೆ (device->coopmat_m == 0) {
                            device->coopmat_acc_f16_support = ನಿಜ;
                            device->coopmat_m = prop.MSize;
                            device->coopmat_n = prop.NSize;
                            device->coopmat_k = prop.KSize;
                        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (device->coopmat_m == prop.MSize && device->coopmat_n == prop.NSize && device->coopmat_k == prop.KSize) {
                            // Only enable if shape is identical
                            device->coopmat_acc_f16_support = ನಿಜ;
                        }
                        ಆದರೆ (prop.MSize == 16 && prop.NSize == 16 && prop.KSize == 16) {
                            device->coopmat_support_16x16x16_f16acc = ನಿಜ;
                        }
                    }
                } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ ((vk::ComponentTypeKHR)prop.AType      == vk::ComponentTypeKHR::eSint8 &&
                           (vk::ComponentTypeKHR)prop.BType      == vk::ComponentTypeKHR::eSint8 &&
                           (vk::ComponentTypeKHR)prop.CType      == vk::ComponentTypeKHR::eSint32 &&
                           (vk::ComponentTypeKHR)prop.ಫಲಿತಾಂಶ_Type == vk::ComponentTypeKHR::eSint32 &&
                           (vk::ScopeKHR)prop.scope == vk::ScopeKHR::eSubgroup &&
                           device->coopmat_int_m == 0
                ) {
                    device->coopmat_int_support = ನಿಜ;
                    device->coopmat_int_m = prop.MSize;
                    device->coopmat_int_n = prop.NSize;
                    device->coopmat_int_k = prop.KSize;
                }
#ಆದರೆ defined(VK_KHR_shader_bfloat16) && defined(GGML_VULKAN_BFLOAT16_GLSLC_SUPPORT)
                ಆದರೆ (bfloat16_support &&
                    prop.AType == VK_COMPONENT_TYPE_BFLOAT16_KHR &&
                    prop.BType == VK_COMPONENT_TYPE_BFLOAT16_KHR &&
                    prop.CType == VK_COMPONENT_TYPE_FLOAT32_KHR &&
                    prop.ಫಲಿತಾಂಶ_Type == VK_COMPONENT_TYPE_FLOAT32_KHR &&
                    (vk::ScopeKHR)prop.scope == vk::ScopeKHR::eSubgroup
                ) {
                    // coopmat sizes not set yet
                    ಆದರೆ (device->coopmat_m == 0) {
                        device->coopmat_bf16_support = ನಿಜ;
                        device->coopmat_m = prop.MSize;
                        device->coopmat_n = prop.NSize;
                        device->coopmat_k = prop.KSize;
                    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (device->coopmat_m == prop.MSize && device->coopmat_n == prop.NSize && device->coopmat_k == prop.KSize) {
                        // Only enable if shape is identical
                        device->coopmat_bf16_support = ನಿಜ;
                    }
                }
#ಆದರೆ_ಅಂತ್ಯ
            }

            ಆದರೆ (device->coopmat_m == 0 || !device->coopmat_acc_f32_support) {
                // No suitable matmul mode found
                GGML_LOG_DEBUG("ggml_vulkan: WARNING: No suitable matrix core mode found. Disabling matrix cores.\n");
                device->coopmat_support = ಸುಳ್ಳು;
            }
        }

        ಆದರೆ (device->coopmat_support) {
            device_extensions.push_back("VK_KHR_cooperative_matrix");
        }
#ಆದರೆ_ಅಂತ್ಯ
        device->name = GGML_VK_NAME + std::to_string(idx);

        device_create_info
            .ಗಣ_Flags(vk::DeviceCreateFlags())
            .ಗಣ_ಸರತಿ_Create_Infos(device_ಸರತಿ_create_infos)
            .ಗಣ_PEnabled_Extension_ಹೆಸರುಗಳು(device_extensions);
        device_create_info.ಗಣ_PNext(&device_features2);
        device->device = device->physical_device.createDevice(device_create_info);

        ಆದರೆ (device->device_fault) {
            device->pfn_vkGetDeviceFaultInfoEXT = (PFN_vkGetDeviceFaultInfoEXT)
                vkGetDeviceProcAddr(device->device, "vkGetDeviceFaultInfoEXT");
        }

        // Queues
        device->compute_ಸರತಿ = ggml_vk_create_ಸರತಿ(device, compute_ಸರತಿ_family_ಸೂಚ್ಯಂಕ, 0, { vk::Pipeline_Stage_Flag_ಬಿಟ್ಟುಗಳು::eComputeShader | vk::Pipeline_Stage_Flag_ಬಿಟ್ಟುಗಳು::eTransfer }, ಸುಳ್ಳು);

        // Shaders
        // Disable matmul tile sizes early if performance low or not supported
        ಗಾಗಿ (uint32_t i = 0; i < GGML_TYPE_ಎಣಿಕೆ; ++i) {
            ಆಯ್ಕೆ (device->vendor_id) {
#ವ್ಯಾಖ್ಯಾನಿತವಾಗದಿದ್ದರೆ GGML_VULKAN_ಸತತ_TESTS
            ಸ್ಥಿತಿ VK_VENDOR_ID_AMD:
                device->mul_mat_l[i]    = device->coopmat_support && device->driver_id != vk::DriverId::eAmdProprietary;
                device->mul_mat_m[i]    = ನಿಜ;
                device->mul_mat_s[i]    = ನಿಜ;
                device->mul_mat_id_l[i] = ಸುಳ್ಳು;
                device->mul_mat_id_m[i] = ನಿಜ;
                device->mul_mat_id_s[i] = ನಿಜ;
                ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ VK_VENDOR_ID_INTEL: {
                // Current Windows driver does not expose BF16 support.
                // We only want to use l_warptile if coopmat is available
                ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ use_l_warptile = (i == GGML_TYPE_BF16) ? (device->coopmat_bf16_support && device->coopmat_support) : device->coopmat_support;
                device->mul_mat_l[i] = use_l_warptile;
                device->mul_mat_id_l[i] = use_l_warptile;
                device->mul_mat_m[i] = ನಿಜ;
                device->mul_mat_s[i] = ನಿಜ;
                device->mul_mat_id_m[i] = ನಿಜ;
                device->mul_mat_id_s[i] = ನಿಜ;
                ನಿಲ್ಲಿಸು;
            }
            ಸ್ಥಿತಿ VK_VENDOR_ID_APPLE:
                device->mul_mat_l[i] = ಸುಳ್ಳು;
                device->mul_mat_m[i] = ನಿಜ;
                device->mul_mat_s[i] = ಸುಳ್ಳು;
                device->mul_mat_id_l[i] = ಸುಳ್ಳು;
                device->mul_mat_id_m[i] = ನಿಜ;
                device->mul_mat_id_s[i] = ಸುಳ್ಳು;
                ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ VK_VENDOR_ID_QUALCOMM:
                device->mul_mat_l[i] = ಸುಳ್ಳು;
                device->mul_mat_m[i] = ನಿಜ;
                device->mul_mat_s[i] = !device->coopmat_support;
                device->mul_mat_id_l[i] = ಸುಳ್ಳು;
                device->mul_mat_id_m[i] = ನಿಜ;
                device->mul_mat_id_s[i] = !device->coopmat_support;
                ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ VK_VENDOR_ID_SAMSUNG:
                device->mul_mat_l[i] = ಸುಳ್ಳು;
                device->mul_mat_m[i] = ನಿಜ;
                device->mul_mat_s[i] = ನಿಜ;
                device->mul_mat_id_l[i] = ಸುಳ್ಳು;
                device->mul_mat_id_m[i] = ನಿಜ;
                device->mul_mat_id_s[i] = ನಿಜ;
                ನಿಲ್ಲಿಸು;
#ಆದರೆ_ಅಂತ್ಯ
            ಪೂರ್ವನಿಯೋಜಿತ:
                device->mul_mat_l[i] = ನಿಜ;
                device->mul_mat_m[i] = ನಿಜ;
                device->mul_mat_s[i] = ನಿಜ;
                device->mul_mat_id_l[i] = ನಿಜ;
                device->mul_mat_id_m[i] = ನಿಜ;
                device->mul_mat_id_s[i] = ನಿಜ;
                ನಿಲ್ಲಿಸು;
            }

#ಆದರೆ VK_ಶೀರ್ಷಿಕೆ_ಆವೃತ್ತಿ >= 287
            // Honeykrisp driver for Asahi Linux doesn't report VK_VENDOR_ID_APPLE.
            // Check for Honeykrisp driver and force same configuration as the VK_VENDOR_ID_APPLE case.
            ಆದರೆ (device->driver_id == vk::DriverId::eMesaHoneykrisp) {
                device->mul_mat_l[i] = ಸುಳ್ಳು;
                device->mul_mat_m[i] = ನಿಜ;
                device->mul_mat_s[i] = ಸುಳ್ಳು;
                device->mul_mat_id_l[i] = ಸುಳ್ಳು;
                device->mul_mat_id_m[i] = ನಿಜ;
                device->mul_mat_id_s[i] = ಸುಳ್ಳು;
            }
#ಆದರೆ_ಅಂತ್ಯ

            device->mul_mat_l_int[i]    = device->mul_mat_l[i];
            device->mul_mat_m_int[i]    = device->mul_mat_m[i];
            device->mul_mat_s_int[i]    = device->mul_mat_s[i];
            device->mul_mat_id_l_int[i] = device->mul_mat_id_l[i];
            device->mul_mat_id_m_int[i] = device->mul_mat_id_m[i];
            device->mul_mat_id_s_int[i] = device->mul_mat_id_s[i];
        }


        std::vector<vk::Descriptor_ಗಣ_Layout_Binding> dsl_binding;
        std::vector<vk::DescriptorBindingFlags> dsl_binding_flags;
        ಗಾಗಿ (uint32_t i = 0; i < MAX_PARAMETER_ಎಣಿಕೆ; i++) {
            dsl_binding.push_back({i, vk::DescriptorType::e_Storage_ಬಫರ್, 1, vk::Shader_Stage_Flag_ಬಿಟ್ಟುಗಳು::eCompute});
            dsl_binding_flags.push_back({});
        }

        vk::Descriptor_ಗಣ_Layout_Binding_Flags_Create_Info dslbfci = { dsl_binding_flags };

        vk::Descriptor_ಗಣ_Layout_Create_Info descriptor_ಗಣ_layout_create_info(
            {},
            dsl_binding);
        descriptor_ಗಣ_layout_create_info.ಗಣ_PNext(&dslbfci);
        device->dsl = device->device.create_Descriptor_ಗಣ_Layout(descriptor_ಗಣ_layout_create_info);

        ggml_vk_ತುಂಬು_shaders(device);

        // Prefer a dedicated transfer queue on AMD dGPUs (non-GCN) when graphics queue use is disabled.
        ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ prefers_transfer_ಸರತಿ =
            device->vendor_id == VK_VENDOR_ID_AMD &&
            device->architecture != AMD_GCN &&
            !device->uma &&
            !allow_graphics_ಸರತಿ;

        ಆದರೆ (!device->single_ಸರತಿ) {
            ಸ್ಥಿರಾಂಕ uint32_t transfer_ಸರತಿ_ಸೂಚ್ಯಂಕ = compute_ಸರತಿ_family_ಸೂಚ್ಯಂಕ == transfer_ಸರತಿ_family_ಸೂಚ್ಯಂಕ ? 1 : 0;
            device->transfer_ಸರತಿ = ggml_vk_create_ಸರತಿ(device, transfer_ಸರತಿ_family_ಸೂಚ್ಯಂಕ, transfer_ಸರತಿ_ಸೂಚ್ಯಂಕ, { vk::Pipeline_Stage_Flag_ಬಿಟ್ಟುಗಳು::eTransfer }, ನಿಜ);

            device->async_use_transfer_ಸರತಿ = prefers_transfer_ಸರತಿ || (getenv("GGML_VK_ASYNC_USE_TRANSFER_QUEUE") != ಶೂನ್ಯಸೂಚಕ);
        } ಇಲ್ಲದಿದ್ದರೆ {
            device->transfer_ಸರತಿ = ggml_vk_create_aliased_ಸರತಿ(device, device->compute_ಸರತಿ);

            device->async_use_transfer_ಸರತಿ = ಸುಳ್ಳು;
        }

        device->fence = device->device.createFence({});

        device->ಬಫರ್_type = {
            /* .iface    = */ ggml_backend_vk_ಬಫರ್_type_interface,
            /* .device   = */ ggml_backend_reg_dev_get(ggml_backend_vk_reg(), idx),
            /* .context  = */ ಹೊಸ ggml_backend_vk_ಬಫರ್_type_context{ device->name, device },
        };

        device->idx = idx;

        device->serialize_submissions = getenv("GGML_VK_SERIALIZE_SUBMISSIONS") != ಶೂನ್ಯಸೂಚಕ;

        device->disable_fusion = getenv("GGML_VK_DISABLE_FUSION") != ಶೂನ್ಯಸೂಚಕ;

        device->disable_descriptor_reuse = getenv("GGML_VK_DISABLE_DESCRIPTOR_REUSE") != ಶೂನ್ಯಸೂಚಕ;

        device->ಕೂಡಿಸು_rms_fusion = !device->disable_fusion &&
                                 device->subgroup_arithmetic &&
                                 device->vendor_id != VK_VENDOR_ID_INTEL;
        device->partials_binding_alignment =
            std::max(4u, (uint32_t)device->properties.limits.min_Storage_ಬಫರ್_ಸ್ಥಾನಾಂತರ_Alignment);

        device->mmvq_ಬಹುಲಕ = 0;
        ಆದರೆ (getenv("GGML_VK_DISABLE_MMVQ")) {
            device->mmvq_ಬಹುಲಕ = -1;
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (getenv("GGML_VK_FORCE_MMVQ")) {
            device->mmvq_ಬಹುಲಕ = 1;
        }

        // publish the device only when it is fully built
        vk_instance.devices[idx] = device;

        ಹಿಂತಿರುಗಿ device;
    }

    ಹಿಂತಿರುಗಿ vk_instance.devices[idx];
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_vk_print_gpu_info(size_t idx) {
    GGML_ASSERT(idx < vk_instance.device_indices.size());
    size_t dev_ಅಂಕೆ = vk_instance.device_indices[idx];
    VK_LOG_DEBUG("ggml_vk_print_gpu_info(" << dev_ಅಂಕೆ << ")");
    GGML_ASSERT(vk_instance_initialized);

    std::vector<vk::PhysicalDevice> devices = vk_instance.instance.enumeratePhysicalDevices();

    ಆದರೆ (dev_ಅಂಕೆ >= devices.size()) {
        std::cerr << "ggml_vulkan: Device with index " << dev_ಅಂಕೆ << " does not exist." << std::ಸಾಲು_ಅಂತ್ಯ;
        ಎಸೆ std::runtime_error("Device not found");
    }

    vk::PhysicalDevice physical_device = devices[dev_ಅಂಕೆ];
    std::vector<vk::ExtensionProperties> ext_props = physical_device.enumerateDeviceExtensionProperties();

    ತಾರ್ಕಿಕ fp16_storage = ಸುಳ್ಳು;
    ತಾರ್ಕಿಕ fp16_compute = ಸುಳ್ಳು;
    ತಾರ್ಕಿಕ coopmat_support = ಸುಳ್ಳು;
    ತಾರ್ಕಿಕ coopmat2_support = ಸುಳ್ಳು;
    ತಾರ್ಕಿಕ coopmat2_decode_ಸದಿಶ_support = ಸುಳ್ಳು;
    ತಾರ್ಕಿಕ integer_dot_ಗುಣಲಬ್ಧ = ಸುಳ್ಳು;
    ತಾರ್ಕಿಕ bfloat16_support = ಸುಳ್ಳು;
    ತಾರ್ಕಿಕ dot2_f16_support = ಸುಳ್ಳು;
    ತಾರ್ಕಿಕ ocp_microscaling_extension = ಸುಳ್ಳು;
    ತಾರ್ಕಿಕ shader_float8_extension = ಸುಳ್ಳು;

    ಗಾಗಿ (ಸ್ವಯಂಚಾಲಿತ properties : ext_props) {
        ಆದರೆ (strcmp("VK_KHR_16bit_storage", properties.extensionName) == 0) {
            fp16_storage = ನಿಜ;
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (strcmp("VK_KHR_shader_float16_int8", properties.extensionName) == 0) {
            fp16_compute = ನಿಜ;
#ಆದರೆ defined(GGML_VULKAN_COOPMAT_GLSLC_SUPPORT)
       } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (strcmp("VK_KHR_cooperative_matrix", properties.extensionName) == 0 &&
                   !getenv("GGML_VK_DISABLE_COOPMAT")) {
            coopmat_support = ನಿಜ;
#ಆದರೆ_ಅಂತ್ಯ
#ಆದರೆ defined(GGML_VULKAN_COOPMAT2_GLSLC_SUPPORT)
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (strcmp("VK_NV_cooperative_matrix2", properties.extensionName) == 0 &&
                   !getenv("GGML_VK_DISABLE_COOPMAT2")) {
            coopmat2_support = ನಿಜ;
#ಆದರೆ_ಅಂತ್ಯ
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (strcmp(VK_NV_COOPERATIVE_ಮಾತೃಕೆ_DECODE_ಸದಿಶ_EXTENSION_NAME, properties.extensionName) == 0 &&
                   !getenv("GGML_VK_DISABLE_COOPMAT2_DECODE_VECTOR")) {
            coopmat2_decode_ಸದಿಶ_support = ನಿಜ;
#ಆದರೆ defined(GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT)
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (strcmp("VK_KHR_shader_integer_dot_product", properties.extensionName) == 0 &&
                    !getenv("GGML_VK_DISABLE_INTEGER_DOT_PRODUCT")) {
            integer_dot_ಗುಣಲಬ್ಧ = ನಿಜ;
#ಆದರೆ_ಅಂತ್ಯ
#ಆದರೆ defined(GGML_VULKAN_BFLOAT16_GLSLC_SUPPORT)
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (strcmp("VK_KHR_shader_bfloat16", properties.extensionName) == 0 &&
                    !getenv("GGML_VK_DISABLE_BFLOAT16")) {
            bfloat16_support = ನಿಜ;
#ಆದರೆ_ಅಂತ್ಯ
#ಆದರೆ defined(GGML_VULKAN_FLOAT_E2M1_GLSLC_SUPPORT)
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (strcmp(VK_EXT_SHADER_OCP_MICROSCALING_TYPES_EXTENSION_NAME, properties.extensionName) == 0) {
            ocp_microscaling_extension = ನಿಜ;
#ಆದರೆ_ಅಂತ್ಯ
#ಆದರೆ defined(GGML_VULKAN_FLOAT_E4M3_GLSLC_SUPPORT)
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (strcmp(VK_EXT_SHADER_FLOAT8_EXTENSION_NAME, properties.extensionName) == 0) {
            shader_float8_extension = ನಿಜ;
#ಆದರೆ_ಅಂತ್ಯ
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (strcmp("VK_VALVE_shader_mixed_float_dot_product", properties.extensionName) == 0 &&
                    !getenv("GGML_VK_DISABLE_DOT2")) {
            dot2_f16_support = ನಿಜ;
        }
    }

    ಸ್ಥಿರಾಂಕ vk_device_architecture device_architecture = get_device_architecture(physical_device);

    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ* GGML_VK_DISABLE_F16 = getenv("GGML_VK_DISABLE_F16");
    ತಾರ್ಕಿಕ force_disable_f16 = GGML_VK_DISABLE_F16 != ಶೂನ್ಯಸೂಚಕ;

    ತಾರ್ಕಿಕ fp16 = !force_disable_f16 && fp16_storage && fp16_compute;

    vk::PhysicalDeviceProperties2 props2;
    vk::PhysicalDeviceMaintenance3Properties props3;
    vk::PhysicalDeviceSubgroupProperties subgroup_props;
    vk::PhysicalDeviceDriverProperties driver_props;
    vk::Physical_Device_Shader_Integer_Dot_ಗುಣಲಬ್ಧ_Properties_KHR shader_integer_dot_ಗುಣಲಬ್ಧ_props;
    props2.p_ಮುಂದಿನ = &props3;
    props3.p_ಮುಂದಿನ = &subgroup_props;
    subgroup_props.p_ಮುಂದಿನ = &driver_props;

    // Pointer to the last chain element
    Vk_ತಳ_ಹೊರಹೋಗು_Structure * ಕೊನೆಯ_struct = (Vk_ತಳ_ಹೊರಹೋಗು_Structure *)&driver_props;

    ಆದರೆ (integer_dot_ಗುಣಲಬ್ಧ) {
        ಕೊನೆಯ_struct->p_ಮುಂದಿನ = (Vk_ತಳ_ಹೊರಹೋಗು_Structure *)&shader_integer_dot_ಗುಣಲಬ್ಧ_props;
        ಕೊನೆಯ_struct = (Vk_ತಳ_ಹೊರಹೋಗು_Structure *)&shader_integer_dot_ಗುಣಲಬ್ಧ_props;
    }

    physical_device.getProperties2(&props2);

    VkPhysicalDeviceFeatures2 device_features2;
    device_features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    device_features2.p_ಮುಂದಿನ = ಶೂನ್ಯಸೂಚಕ;

    VkPhysicalDeviceVulkan11Features vk11_features;
    vk11_features.p_ಮುಂದಿನ = ಶೂನ್ಯಸೂಚಕ;
    vk11_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
    device_features2.p_ಮುಂದಿನ = &vk11_features;

    VkPhysicalDeviceVulkan12Features vk12_features;
    vk12_features.p_ಮುಂದಿನ = ಶೂನ್ಯಸೂಚಕ;
    vk12_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    vk11_features.p_ಮುಂದಿನ = &vk12_features;

    // Pointer to the last chain element
    ಕೊನೆಯ_struct = (Vk_ತಳ_ಹೊರಹೋಗು_Structure *)&vk12_features;

#ಆದರೆ defined(GGML_VULKAN_COOPMAT_GLSLC_SUPPORT)
    Vk_Physical_Device_Cooperative_ಮಾತೃಕೆ_Features_KHR coopmat_features;
    coopmat_features.p_ಮುಂದಿನ = ಶೂನ್ಯಸೂಚಕ;
    coopmat_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_ಮಾತೃಕೆ_FEATURES_KHR;
    coopmat_features.cooperative_ಮಾತೃಕೆ_2 = VK_FALSE;

    ಆದರೆ (coopmat_support) {
        ಕೊನೆಯ_struct->p_ಮುಂದಿನ = (Vk_ತಳ_ಹೊರಹೋಗು_Structure *)&coopmat_features;
        ಕೊನೆಯ_struct = (Vk_ತಳ_ಹೊರಹೋಗು_Structure *)&coopmat_features;
    }
#ಆದರೆ_ಅಂತ್ಯ

    Vk_Physical_Device_Shader_Integer_Dot_ಗುಣಲಬ್ಧ_Features_KHR shader_integer_dot_ಗುಣಲಬ್ಧ_features {};
    shader_integer_dot_ಗುಣಲಬ್ಧ_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_ಗುಣಲಬ್ಧ_FEATURES_KHR;
    ಆದರೆ (integer_dot_ಗುಣಲಬ್ಧ) {
        ಕೊನೆಯ_struct->p_ಮುಂದಿನ = (Vk_ತಳ_ಹೊರಹೋಗು_Structure *)&shader_integer_dot_ಗುಣಲಬ್ಧ_features;
        ಕೊನೆಯ_struct = (Vk_ತಳ_ಹೊರಹೋಗು_Structure *)&shader_integer_dot_ಗುಣಲಬ್ಧ_features;
    }

#ಆದರೆ defined(VK_KHR_shader_bfloat16)
    VkPhysicalDeviceShaderBfloat16FeaturesKHR bfloat16_features {};
    bfloat16_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_BFLOAT16_FEATURES_KHR;
    ಆದರೆ (bfloat16_support) {
        ಕೊನೆಯ_struct->p_ಮುಂದಿನ = (Vk_ತಳ_ಹೊರಹೋಗು_Structure *)&bfloat16_features;
        ಕೊನೆಯ_struct = (Vk_ತಳ_ಹೊರಹೋಗು_Structure *)&bfloat16_features;
    }
#ಆದರೆ_ಅಂತ್ಯ

#ಆದರೆ defined(VK_NV_cooperative_matrix2)
    VkPhysicalDeviceCooperativeMatrix2FeaturesNV coopmat2_features {};
    coopmat2_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_ಮಾತೃಕೆ_2_FEATURES_NV;
    ಆದರೆ (coopmat2_support) {
        ಕೊನೆಯ_struct->p_ಮುಂದಿನ = (Vk_ತಳ_ಹೊರಹೋಗು_Structure *)&coopmat2_features;
        ಕೊನೆಯ_struct = (Vk_ತಳ_ಹೊರಹೋಗು_Structure *)&coopmat2_features;
    }
#ಆದರೆ_ಅಂತ್ಯ

    Vk_Physical_Device_Cooperative_ಮಾತೃಕೆ_Decode_ಸದಿಶ_Features_NV coopmat2_decode_ಸದಿಶ_features {};
    coopmat2_decode_ಸದಿಶ_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_ಮಾತೃಕೆ_DECODE_ಸದಿಶ_FEATURES_NV;
    ಆದರೆ (coopmat2_decode_ಸದಿಶ_support) {
        ಕೊನೆಯ_struct->p_ಮುಂದಿನ = (Vk_ತಳ_ಹೊರಹೋಗು_Structure *)&coopmat2_decode_ಸದಿಶ_features;
        ಕೊನೆಯ_struct = (Vk_ತಳ_ಹೊರಹೋಗು_Structure *)&coopmat2_decode_ಸದಿಶ_features;
    }

    Vk_Physical_Device_Shader_Mixed_Float_Dot_ಗುಣಲಬ್ಧ_Features_VALVE dot2_features {};
    dot2_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_MIXED_FLOAT_DOT_ಗುಣಲಬ್ಧ_FEATURES_VALVE;
    ಆದರೆ (dot2_f16_support) {
        ಕೊನೆಯ_struct->p_ಮುಂದಿನ = (Vk_ತಳ_ಹೊರಹೋಗು_Structure *)&dot2_features;
        ಕೊನೆಯ_struct = (Vk_ತಳ_ಹೊರಹೋಗು_Structure *)&dot2_features;
    }

#ಆದರೆ defined(GGML_VULKAN_FLOAT_E2M1_GLSLC_SUPPORT) && defined(GGML_VULKAN_FLOAT_E4M3_GLSLC_SUPPORT)
    VkPhysicalDeviceShaderOCPMicroscalingTypesFeaturesEXT ocp_microscaling_features {};
    ocp_microscaling_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_OCP_MICROSCALING_TYPES_FEATURES_EXT;
    VkPhysicalDeviceShaderFloat8FeaturesEXT shader_float8_features {};
    shader_float8_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT8_FEATURES_EXT;
    ಆದರೆ (ocp_microscaling_extension) {
        ಕೊನೆಯ_struct->p_ಮುಂದಿನ = (Vk_ತಳ_ಹೊರಹೋಗು_Structure *)&ocp_microscaling_features;
        ಕೊನೆಯ_struct = (Vk_ತಳ_ಹೊರಹೋಗು_Structure *)&ocp_microscaling_features;
    }
    ಆದರೆ (shader_float8_extension) {
        ಕೊನೆಯ_struct->p_ಮುಂದಿನ = (Vk_ತಳ_ಹೊರಹೋಗು_Structure *)&shader_float8_features;
        ಕೊನೆಯ_struct = (Vk_ತಳ_ಹೊರಹೋಗು_Structure *)&shader_float8_features;
    }
#ಆದರೆ_ಅಂತ್ಯ

    vkGetPhysicalDeviceFeatures2(physical_device, &device_features2);

    fp16 = fp16 && vk12_features.shaderFloat16;

#ಆದರೆ defined(VK_KHR_shader_bfloat16)
    ತಾರ್ಕಿಕ bf16 = bfloat16_support && bfloat16_features.shaderBFloat16Type;
#ಇಲ್ಲದಿದ್ದರೆ
    ತಾರ್ಕಿಕ bf16 = ಸುಳ್ಳು;
#ಆದರೆ_ಅಂತ್ಯ

    uint32_t default_subgroup_ಗಾತ್ರ = get_subgroup_ಗಾತ್ರ("", device_architecture);
    ಸ್ಥಿರಾಂಕ size_t subgroup_ಗಾತ್ರ = (default_subgroup_ಗಾತ್ರ != 0) ? default_subgroup_ಗಾತ್ರ : subgroup_props.subgroup_ಗಾತ್ರ_2;
    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ uma = props2.properties.deviceType == vk::PhysicalDeviceType::eIntegratedGpu;

    integer_dot_ಗುಣಲಬ್ಧ = integer_dot_ಗುಣಲಬ್ಧ
                       && shader_integer_dot_ಗುಣಲಬ್ಧ_props.integer_Dot_Product4x8_ಬಿಟ್_Packed_Signed_Accelerated
                       && shader_integer_dot_ಗುಣಲಬ್ಧ_features.shader_Integer_Dot_ಗುಣಲಬ್ಧ;

    coopmat_support = coopmat_support
#ಆದರೆ defined(GGML_VULKAN_COOPMAT_GLSLC_SUPPORT)
                   && coopmat_features.cooperative_ಮಾತೃಕೆ_2
#ಆದರೆ_ಅಂತ್ಯ
                   && ggml_vk_khr_cooperative_ಮಾತೃಕೆ_support(props2.properties, driver_props, device_architecture);

#ಆದರೆ defined(VK_NV_cooperative_matrix2) && defined(GGML_VULKAN_COOPMAT2_GLSLC_SUPPORT)
    coopmat2_support = coopmat2_support &&
                       coopmat2_features.cooperative_ಮಾತೃಕೆ_Workgroup_Scope &&
                       coopmat2_features.cooperative_ಮಾತೃಕೆ_Flexible_Dimensions &&
                       coopmat2_features.cooperative_ಮಾತೃಕೆ_Reductions &&
                       coopmat2_features.cooperative_ಮಾತೃಕೆ_Conversions &&
                       coopmat2_features.cooperative_ಮಾತೃಕೆ_ಪ್ರತಿ_Element_Operations &&
                       coopmat2_features.cooperative_ಮಾತೃಕೆ_Tensor_Addressing &&
                       coopmat2_features.cooperative_ಮಾತೃಕೆ_ಖಂಡ_Loads;
#ಇಲ್ಲದಿದ್ದರೆ
    coopmat2_support = ಸುಳ್ಳು;
#ಆದರೆ_ಅಂತ್ಯ

    coopmat2_decode_ಸದಿಶ_support = coopmat2_decode_ಸದಿಶ_support && coopmat2_decode_ಸದಿಶ_features.cooperative_ಮಾತೃಕೆ_Decode_ಸದಿಶ;
#ಆದರೆ !defined(GGML_VULKAN_COOPMAT2_DECODE_ಸದಿಶ_GLSLC_SUPPORT)
    coopmat2_decode_ಸದಿಶ_support = ಸುಳ್ಳು;
#ಆದರೆ_ಅಂತ್ಯ

    std::string ಮಾತೃಕೆ_cores = coopmat2_support ? (coopmat2_decode_ಸದಿಶ_support ? "NV_coopmat2v" : "NV_coopmat2")
                             : coopmat_support  ? "KHR_coopmat"
                             : "none";

    ತಾರ್ಕಿಕ dot2_f16 = dot2_f16_support && dot2_features.shader_Mixed_Float_Dot_ಗುಣಲಬ್ಧ_Float16_Acc_Float32;
    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *fp16_str = fp16 ? (dot2_f16 ? "dot2" : "1") : "0";
#ಆದರೆ defined(GGML_VULKAN_FLOAT_E2M1_GLSLC_SUPPORT) && defined(GGML_VULKAN_FLOAT_E4M3_GLSLC_SUPPORT)
    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ fp4 = ocp_microscaling_extension && ocp_microscaling_features.shaderFloat4 &&
                     shader_float8_extension && shader_float8_features.shaderFloat8 &&
                     !getenv("GGML_VK_DISABLE_OCP_FP4");
#ಇಲ್ಲದಿದ್ದರೆ
    GGML_ಬಳಸದ(ocp_microscaling_extension);
    GGML_ಬಳಸದ(shader_float8_extension);
    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ fp4 = ಸುಳ್ಳು;
#ಆದರೆ_ಅಂತ್ಯ

    std::string device_name = props2.properties.deviceName.ದತ್ತಾಂಶ();
    GGML_LOG_DEBUG("ggml_vulkan: %zu = %s (%s) | uma: %d | fp16: %s | bf16: %d | fp4: %d | warp size: %zu | shared memory: %d | int dot: %d | matrix cores: %s\n",
              idx, device_name.c_str(), driver_props.driverName.ದತ್ತಾಂಶ(), uma, fp16_str, bf16, fp4, subgroup_ಗಾತ್ರ,
              props2.properties.limits.max_Compute_Shared_ಸ್ಮರಣೆ_ಗಾತ್ರ, integer_dot_ಗುಣಲಬ್ಧ, ಮಾತೃಕೆ_cores.c_str());

    ಆದರೆ (props2.properties.deviceType == vk::PhysicalDeviceType::eCpu) {
        GGML_LOG_DEBUG("ggml_vulkan: Warning: Device type is CPU. This is probably not the device you want.\n");
    }
}

ಸ್ಥಿರ DispatchLoaderDynamic ggml_vk_default_dispatcher_instance;

DispatchLoaderDynamic & ggml_vk_default_dispatcher() {
    ಹಿಂತಿರುಗಿ ggml_vk_default_dispatcher_instance;
}

ಶೂನ್ಯ ggml_vk_instance_init() {
    ಆದರೆ (vk_instance_initialized) {
        ಹಿಂತಿರುಗಿ;
    }
    VK_LOG_DEBUG("ggml_vk_instance_init()");

    // See https://github.com/KhronosGroup/Vulkan-Hpp?tab=readme-ov-file#extensions--per-device-function-pointers-
    ggml_vk_default_dispatcher_instance.init(vkGetInstanceProcAddr);

    // vkEnumerateInstanceVersion is Vulkan 1.1. A null value indicated Vulkan 1.0.
    ಆದರೆ (ggml_vk_default_dispatcher_instance.vk_Enumerate_Instance_ಆವೃತ್ತಿ == ಶೂನ್ಯಸೂಚಕ) {
        GGML_LOG_ERROR("ggml_vulkan: Error: Vulkan 1.2 required.");
        ಎಸೆ vk::SystemError(vk::ಫಲಿತಾಂಶ_2::eErrorFeatureNotPresent, "Vulkan 1.2 required");
    }

    uint32_t api_ಆವೃತ್ತಿ = vk::enumerate_Instance_ಆವೃತ್ತಿ();

    ಆದರೆ (api_ಆವೃತ್ತಿ < VK_API_ಆವೃತ್ತಿ_1_2) {
        GGML_LOG_ERROR("ggml_vulkan: Error: Vulkan 1.2 required.");
        ಎಸೆ vk::SystemError(vk::ಫಲಿತಾಂಶ_2::eErrorFeatureNotPresent, "Vulkan 1.2 required");
    }

    vk::ApplicationInfo app_info{ "ggml-vulkan", 1, ಶೂನ್ಯಸೂಚಕ, 0, api_ಆವೃತ್ತಿ };

    ಸ್ಥಿರಾಂಕ std::vector<vk::ExtensionProperties> instance_extensions = vk::enumerateInstanceExtensionProperties();
    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ layer_settings = ggml_vk_instance_layer_settings_ಲಭ್ಯ();
#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ __APPLE__
    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ portability_enumeration_ext = ggml_vk_instance_portability_enumeration_ext_ಲಭ್ಯ(instance_extensions);
#ಆದರೆ_ಅಂತ್ಯ
    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ debug_utils_ext = ggml_vk_instance_debug_utils_ext_ಲಭ್ಯ(instance_extensions) && getenv("GGML_VK_DEBUG_MARKERS") != ಶೂನ್ಯಸೂಚಕ;
    std::vector<ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ*> layers;

    ಆದರೆ (layer_settings) {
        layers.push_back("VK_LAYER_KHRONOS_validation");
    }
    std::vector<ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ*> extensions;
    ಆದರೆ (layer_settings) {
        extensions.push_back("VK_EXT_layer_settings");
    }
#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ __APPLE__
    ಆದರೆ (portability_enumeration_ext) {
        extensions.push_back("VK_KHR_portability_enumeration");
    }
#ಆದರೆ_ಅಂತ್ಯ
    ಆದರೆ (debug_utils_ext) {
        extensions.push_back("VK_EXT_debug_utils");
    }
    VkBool32 enable_ಉತ್ತಮ_practice = layer_settings;
    std::vector<vk::LayerSettingEXT> settings = {
        {
            "VK_LAYER_KHRONOS_validation",
            "validate_best_practices",
            vk::LayerSettingTypeEXT::eBool32,
            1,
            &enable_ಉತ್ತಮ_practice
        },
    };
    vk::LayerSettingsCreateInfoEXT layer_setting_info(settings);
    vk::InstanceCreateInfo instance_create_info(vk::InstanceCreateFlags{}, &app_info, layers, extensions, &layer_setting_info);
#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ __APPLE__
    ಆದರೆ (portability_enumeration_ext) {
        instance_create_info.flags |= vk::Instance_Create_Flag_ಬಿಟ್ಟುಗಳು::eEnumeratePortabilityKHR;
    }
#ಆದರೆ_ಅಂತ್ಯ

    vk_instance.instance = vk::createInstance(instance_create_info);
    vk_instance_initialized = ನಿಜ;

    ಆದರೆ (debug_utils_ext) {
        vk_instance.debug_utils_support              = ನಿಜ;
        vk_instance.pfn_vk_ಗಣ_Debug_Utils_Object_Name_EXT = (PFN_vk_ಗಣ_Debug_Utils_Object_Name_EXT) vkGetInstanceProcAddr(vk_instance.instance, "vkSetDebugUtilsObjectNameEXT");
        vk_instance.pfn_vk_ಸರತಿ_Begin_Debug_Utils_Label_EXT = (PFN_vk_ಸರತಿ_Begin_Debug_Utils_Label_EXT) vkGetInstanceProcAddr(vk_instance.instance, "vkQueueBeginDebugUtilsLabelEXT");
        vk_instance.pfn_vk_ಸರತಿ_ಕೊನೆ_Debug_Utils_Label_EXT = (PFN_vk_ಸರತಿ_ಕೊನೆ_Debug_Utils_Label_EXT) vkGetInstanceProcAddr(vk_instance.instance, "vkQueueEndDebugUtilsLabelEXT");
        vk_instance.pfn_vkCmdBeginDebugUtilsLabelEXT = (PFN_vkCmdBeginDebugUtilsLabelEXT) vkGetInstanceProcAddr(vk_instance.instance, "vkCmdBeginDebugUtilsLabelEXT");
        vk_instance.pfn_vk_Cmd_ಕೊನೆ_Debug_Utils_Label_EXT =   (PFN_vk_Cmd_ಕೊನೆ_Debug_Utils_Label_EXT) vkGetInstanceProcAddr(vk_instance.instance, "vkCmdEndDebugUtilsLabelEXT");
        vk_instance.pfn_vk_Cmd_ಸೇರಿಸು_Debug_Utils_Label_EXT = (PFN_vk_Cmd_ಸೇರಿಸು_Debug_Utils_Label_EXT) vkGetInstanceProcAddr(vk_instance.instance, "vkCmdInsertDebugUtilsLabelEXT");
    }

    vk_perf_logger_enabled = getenv("GGML_VK_PERF_LOGGER") != ಶೂನ್ಯಸೂಚಕ;
    vk_perf_logger_concurrent = getenv("GGML_VK_PERF_LOGGER_CONCURRENT") != ಶೂನ್ಯಸೂಚಕ;
    vk_enable_sync_logger = getenv("GGML_VK_SYNC_LOGGER") != ಶೂನ್ಯಸೂಚಕ;
    vk_ಸ್ಮರಣೆ_logger_enabled = getenv("GGML_VK_MEMORY_LOGGER") != ಶೂನ್ಯಸೂಚಕ;
    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ* GGML_VK_PIPELINE_STATS = getenv("GGML_VK_PIPELINE_STATS");
    ಆದರೆ (GGML_VK_PIPELINE_STATS != ಶೂನ್ಯಸೂಚಕ) {
        vk_pipeline_stats_filter = GGML_VK_PIPELINE_STATS;
    }
    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ* GGML_VK_PERF_LOGGER_FREQUENCY = getenv("GGML_VK_PERF_LOGGER_FREQUENCY");

    ಆದರೆ (GGML_VK_PERF_LOGGER_FREQUENCY != ಶೂನ್ಯಸೂಚಕ) {
        vk_perf_logger_frequency = std::stoul(GGML_VK_PERF_LOGGER_FREQUENCY);
    }

    // See https://github.com/KhronosGroup/Vulkan-Hpp?tab=readme-ov-file#extensions--per-device-function-pointers-
    VULKAN_HPP_DEFAULT_DISPATCHER.init(vk_instance.instance);

    std::vector<vk::PhysicalDevice> devices = vk_instance.instance.enumeratePhysicalDevices();

    // Emulate behavior of CUDA_VISIBLE_DEVICES for Vulkan
    ಅಕ್ಷರ * devices_env = getenv("GGML_VK_VISIBLE_DEVICES");
    ಆದರೆ (devices_env != ಶೂನ್ಯಸೂಚಕ) {
        size_t ಅಂಕೆ_ಲಭ್ಯ_devices = devices.size();

        std::string devices(devices_env);
        std::replace(devices.begin(), devices.end(), ',', ' ');

        std::stringstream ss(devices);
        size_t tmp;
        ಆಗ (ss >> tmp) {
            ಆದರೆ(tmp >= ಅಂಕೆ_ಲಭ್ಯ_devices) {
                std::cerr << "ggml_vulkan: Invalid device index " << tmp << " in GGML_VK_VISIBLE_DEVICES." << std::ಸಾಲು_ಅಂತ್ಯ;
                ಎಸೆ std::runtime_error("Invalid Vulkan device index");
            }
            vk_instance.device_indices.push_back(tmp);
        }
    } ಇಲ್ಲದಿದ್ದರೆ {
        // If no vulkan devices are found, return early
        ಆದರೆ (devices.ಖಾಲಿಗಳು()) {
            GGML_LOG_INFO("ggml_vulkan: No devices found.\n");
            ಹಿಂತಿರುಗಿ;
        }

        // Default to using all dedicated GPUs
        ಗಾಗಿ (size_t i = 0; i < devices.size(); i++) {
            vk::PhysicalDeviceProperties2 new_props;
            vk::PhysicalDeviceDriverProperties new_driver;
            vk::PhysicalDeviceIDProperties new_id;
            new_props.p_ಮುಂದಿನ = &new_driver;
            new_driver.p_ಮುಂದಿನ = &new_id;
            devices[i].getProperties2(&new_props);

            ಆದರೆ ((new_props.properties.deviceType == vk::PhysicalDeviceType::eDiscreteGpu || new_props.properties.deviceType == vk::PhysicalDeviceType::eIntegratedGpu) && ggml_vk_device_is_supported(devices[i])) {
                // Check if there are two physical devices corresponding to the same GPU
                // This handles the case where the same GPU appears with different drivers (e.g., RADV + AMDVLK on Linux),
                // see https://github.com/ggml-org/llama.cpp/pull/7582 for original deduplication.
                // MoltenVK on macOS may report the same UUID for distinct GPUs on multi-GPU cards,
                // see https://github.com/KhronosGroup/MoltenVK/issues/2683. Skip when both old/new
                // driver is MoltenVK
                ಸ್ವಯಂಚಾಲಿತ old_device = std::find_if(
                    vk_instance.device_indices.begin(),
                    vk_instance.device_indices.end(),
                    [&devices, &new_id, &new_driver](ಸ್ಥಿರಾಂಕ size_t k){
                        vk::PhysicalDeviceProperties2 old_props;
                        vk::PhysicalDeviceDriverProperties old_driver;
                        vk::PhysicalDeviceIDProperties old_id;
                        old_props.p_ಮುಂದಿನ = &old_driver;
                        old_driver.p_ಮುಂದಿನ = &old_id;
                        devices[k].getProperties2(&old_props);

                        ತಾರ್ಕಿಕ same_uuid = std::equal(std::begin(old_id.deviceUUID), std::end(old_id.deviceUUID), std::begin(new_id.deviceUUID));
                        same_uuid = same_uuid || (
                            old_id.deviceLUIDValid && new_id.deviceLUIDValid &&
                            std::equal(std::begin(old_id.deviceLUID), std::end(old_id.deviceLUID), std::begin(new_id.deviceLUID))
                        );
                        ತಾರ್ಕಿಕ both_molten_vk = (new_driver.driverID == vk::DriverId::eMoltenvk && old_driver.driverID == vk::DriverId::eMoltenvk);

                        ಹಿಂತಿರುಗಿ same_uuid && !both_molten_vk;
                    }
                );
                ಆದರೆ (old_device == vk_instance.device_indices.end()) {
                    vk_instance.device_indices.push_back(i);
                } ಇಲ್ಲದಿದ್ದರೆ {
                    // There can be two physical devices corresponding to the same GPU if there are 2 different drivers
                    // This can cause error when splitting layers aross the devices, need to keep only 1
                    VK_LOG_DEBUG("Device " << i << " and device " << *old_device << " have the same deviceUUID");

                    vk::PhysicalDeviceProperties2 old_props;
                    vk::PhysicalDeviceDriverProperties old_driver;
                    old_props.p_ಮುಂದಿನ = &old_driver;
                    devices[*old_device].getProperties2(&old_props);

                    std::map<vk::DriverId, ಪೂರ್ಣಾಂಕ> driver_priorities {};
                    ಪೂರ್ಣಾಂಕ old_ಆದ್ಯತೆ = std::numeric_limits<ಪೂರ್ಣಾಂಕ>::max();
                    ಪೂರ್ಣಾಂಕ new_ಆದ್ಯತೆ = std::numeric_limits<ಪೂರ್ಣಾಂಕ>::max();

                    // Check https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkDriverId.html for the list of driver id
                    // Smaller number -> higher priority
                    ಆಯ್ಕೆ (old_props.properties.vendorID) {
                        ಸ್ಥಿತಿ VK_VENDOR_ID_AMD:
                            driver_priorities[vk::DriverId::eMesaRadv] = 1;
                            driver_priorities[vk::DriverId::e_Amd_Open_ಮೂಲ] = 2;
                            driver_priorities[vk::DriverId::eAmdProprietary] = 3;
                            ನಿಲ್ಲಿಸು;
                        ಸ್ಥಿತಿ VK_VENDOR_ID_INTEL:
                            driver_priorities[vk::DriverId::e_Intel_Open_ಮೂಲ_MESA] = 1;
                            driver_priorities[vk::DriverId::eIntelProprietaryWindows] = 2;
                            ನಿಲ್ಲಿಸು;
                        ಸ್ಥಿತಿ VK_VENDOR_ID_NVIDIA:
                            driver_priorities[vk::DriverId::eNvidiaProprietary] = 1;
#ಆದರೆ defined(VK_API_ಆವೃತ್ತಿ_1_3) && VK_ಶೀರ್ಷಿಕೆ_ಆವೃತ್ತಿ >= 235
                            driver_priorities[vk::DriverId::eMesaNvk] = 2;
#ಆದರೆ_ಅಂತ್ಯ
                            ನಿಲ್ಲಿಸು;
                        ಸ್ಥಿತಿ VK_VENDOR_ID_QUALCOMM:
                            driver_priorities[vk::DriverId::eQualcommProprietary] = 1;
                            driver_priorities[vk::DriverId::eMesaTurnip] = 2;
                            ನಿಲ್ಲಿಸು;
                    }
                    driver_priorities[vk::DriverId::eMesaDozen] = 100;

                    ಆದರೆ (driver_priorities.count(old_driver.driverID)) {
                        old_ಆದ್ಯತೆ = driver_priorities[old_driver.driverID];
                    }
                    ಆದರೆ (driver_priorities.count(new_driver.driverID)) {
                        new_ಆದ್ಯತೆ = driver_priorities[new_driver.driverID];
                    }

                    ಆದರೆ (new_ಆದ್ಯತೆ < old_ಆದ್ಯತೆ) {
                        ಸ್ವಯಂಚಾಲಿತ r = std::remove(vk_instance.device_indices.begin(), vk_instance.device_indices.end(), *old_device);
                        vk_instance.device_indices.erase(r, vk_instance.device_indices.end());
                        vk_instance.device_indices.push_back(i);

                        VK_LOG_DEBUG("Prioritize device " << i << " driver " << new_driver.driverName << " over device " << *old_device << " driver " << old_driver.driverName);
                    }
                    ಇಲ್ಲದಿದ್ದರೆ {
                        VK_LOG_DEBUG("Prioritize device " << *old_device << " driver " << old_driver.driverName << " over device " << i << " driver " << new_driver.driverName << std::ಸಾಲು_ಅಂತ್ಯ);
                    }
                }
            }
        }

        // If no GPUs found, fall back to the first non-CPU device.
        // If only CPU devices are available, return without devices.
        ಆದರೆ (vk_instance.device_indices.ಖಾಲಿಗಳು()) {
            ಗಾಗಿ (size_t i = 0; i < devices.size(); i++) {
                ಆದರೆ (devices[i].getProperties().deviceType != vk::PhysicalDeviceType::eCpu) {
                    vk_instance.device_indices.push_back(i);
                    ನಿಲ್ಲಿಸು;
                }
            }
        }

        ಆದರೆ (vk_instance.device_indices.ಖಾಲಿಗಳು()) {
            GGML_LOG_INFO("ggml_vulkan: No devices found.\n");
            ಹಿಂತಿರುಗಿ;
        }
    }
    GGML_LOG_DEBUG("ggml_vulkan: Found %zu Vulkan devices:\n", vk_instance.device_indices.size());

    size_t host_ಬಫರ್_import_ಎಣಿಕೆ = 0;
    ಗಾಗಿ (size_t i = 0; i < vk_instance.device_indices.size(); i++) {
        vk::PhysicalDevice vkdev = devices[vk_instance.device_indices[i]];
        std::vector<vk::ExtensionProperties> extensionprops = vkdev.enumerateDeviceExtensionProperties();

        ತಾರ್ಕಿಕ membudget_supported = ಸುಳ್ಳು;
        ತಾರ್ಕಿಕ external_ಸ್ಮರಣೆ_host_supported = ಸುಳ್ಳು;
        ಗಾಗಿ (ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ & ext : extensionprops) {
            ಆದರೆ (strcmp(VK_EXT_ಸ್ಮರಣೆ_BUDGET_EXTENSION_NAME, ext.extensionName) == 0) {
                membudget_supported = ನಿಜ;
            } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (strcmp("VK_EXT_external_memory_host", ext.extensionName) == 0) {
                external_ಸ್ಮರಣೆ_host_supported = ನಿಜ;
            }
        }

        vk_instance.device_supports_membudget.push_back(membudget_supported);

        ತಾರ್ಕಿಕ host_ಬಫರ್_import = ಸುಳ್ಳು;
        ಆದರೆ (external_ಸ್ಮರಣೆ_host_supported) {
            vk::PhysicalDeviceProperties2 props2;
            vk::PhysicalDeviceDriverProperties driver_props;
            vk::Physical_Device_External_ಸ್ಮರಣೆ_Host_Properties_EXT ext_props;
            props2.p_ಮುಂದಿನ = &driver_props;
            driver_props.p_ಮುಂದಿನ = &ext_props;
            vkdev.getProperties2(&props2);
            host_ಬಫರ್_import = ggml_vk_external_ಸ್ಮರಣೆ_host_usable(driver_props.driverID) &&
                                 ggml_vk_host_ಬಫರ್_import_usable(props2.properties.vendorID);
            ಆದರೆ (host_ಬಫರ್_import) {
                vk_instance.host_import_alignment = std::max<size_t>(vk_instance.host_import_alignment, ext_props.min_Imported_Host_ಸೂಚಕ_Alignment);
                host_ಬಫರ್_import_ಎಣಿಕೆ++;
            }
        }
        vk_instance.device_host_ಬಫರ್_import.push_back(host_ಬಫರ್_import);

        ggml_vk_print_gpu_info(i);
    }

    vk_instance.multi_device = host_ಬಫರ್_import_ಎಣಿಕೆ > 1 && vk_instance.device_host_ಬಫರ್_import[0];
}

ಶೂನ್ಯ ggml_vk_init(ggml_backend_vk_context * ctx, size_t idx) {
    VK_LOG_DEBUG("ggml_vk_init(" << ctx->name << ", " << idx << ")");
    ggml_vk_instance_init();
    GGML_ASSERT(idx < vk_instance.device_indices.size());

    ctx->name = GGML_VK_NAME + std::to_string(idx);

    ctx->device = ggml_vk_get_device(idx);

    ctx->semaphore_idx = 0;
    ctx->event_idx = 0;

    ctx->prealloc_ಗಾತ್ರ_x = 0;
    ctx->prealloc_ಗಾತ್ರ_y = 0;
    ctx->prealloc_ಗಾತ್ರ_ವಿಭಜಿಸು_k = 0;
    // Fixed size of 1KB, for deterministic behavior
    ctx->prealloc_ಗಾತ್ರ_ಕೂಡಿಸು_rms_partials = 1024;

    ctx->fence = ctx->device->device.createFence({});
    ctx->almost_ready_fence = ctx->device->device.createFence({});

    ctx->compute_cmd_pool.init(ctx->device, ctx->device->compute_ಸರತಿ.get());
    ಆದರೆ (ctx->device->async_use_transfer_ಸರತಿ) {
        vk::SemaphoreTypeCreateInfo tci{ vk::SemaphoreType::eTimeline, 0 };
        vk::SemaphoreCreateInfo ci{};
        ci.ಗಣ_PNext(&tci);
        ctx->transfer_semaphore.s = ctx->device->device.createSemaphore(ci);
        ctx->transfer_semaphore.value = 0;

        ctx->transfer_cmd_pool.init(ctx->device, ctx->device->transfer_ಸರತಿ.get());
    }

    ಆದರೆ (vk_perf_logger_enabled) {
        ctx->perf_logger = std::unique_ptr<vk_perf_logger>(ಹೊಸ vk_perf_logger());
    }

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_VULKAN_CHECK_RESULTS
    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ* skip_checks = getenv("GGML_VULKAN_SKIP_CHECKS");
    vk_skip_checks = (skip_checks == ಖಾಲಿ ? 0 : atoi(skip_checks));
    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ* output_tensor = getenv("GGML_VULKAN_OUTPUT_TENSOR");
    vk_output_tensor = (output_tensor == ಖಾಲಿ ? 0 : atoi(output_tensor));
#ಆದರೆ_ಅಂತ್ಯ
}

vk_pipeline ggml_vk_get_ಗಮ್ಯ_fp16(ggml_backend_vk_context * ctx, ggml_type type) {
    VK_LOG_DEBUG("ggml_vk_get_to_fp16()");
    ಆಯ್ಕೆ (type) {
        ಸ್ಥಿತಿ GGML_TYPE_F32:
        ಸ್ಥಿತಿ GGML_TYPE_Q1_0:
        ಸ್ಥಿತಿ GGML_TYPE_Q2_0:
        ಸ್ಥಿತಿ GGML_TYPE_Q4_0:
        ಸ್ಥಿತಿ GGML_TYPE_Q4_1:
        ಸ್ಥಿತಿ GGML_TYPE_Q5_0:
        ಸ್ಥಿತಿ GGML_TYPE_Q5_1:
        ಸ್ಥಿತಿ GGML_TYPE_Q8_0:
        ಸ್ಥಿತಿ GGML_TYPE_Q2_K:
        ಸ್ಥಿತಿ GGML_TYPE_Q3_K:
        ಸ್ಥಿತಿ GGML_TYPE_Q4_K:
        ಸ್ಥಿತಿ GGML_TYPE_Q5_K:
        ಸ್ಥಿತಿ GGML_TYPE_Q6_K:
        ಸ್ಥಿತಿ GGML_TYPE_IQ1_S:
        ಸ್ಥಿತಿ GGML_TYPE_IQ1_M:
        ಸ್ಥಿತಿ GGML_TYPE_IQ2_XXS:
        ಸ್ಥಿತಿ GGML_TYPE_IQ2_XS:
        ಸ್ಥಿತಿ GGML_TYPE_IQ2_S:
        ಸ್ಥಿತಿ GGML_TYPE_IQ3_XXS:
        ಸ್ಥಿತಿ GGML_TYPE_IQ3_S:
        ಸ್ಥಿತಿ GGML_TYPE_IQ4_XS:
        ಸ್ಥಿತಿ GGML_TYPE_IQ4_NL:
        ಸ್ಥಿತಿ GGML_TYPE_MXFP4:
        ಸ್ಥಿತಿ GGML_TYPE_NVFP4:
        ಸ್ಥಿತಿ GGML_TYPE_TQ1_0:
        ಸ್ಥಿತಿ GGML_TYPE_TQ2_0:
            ನಿಲ್ಲಿಸು;
        ಪೂರ್ವನಿಯೋಜಿತ:
            ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
    }

    ಹಿಂತಿರುಗಿ ctx->device->pipeline_dequant[type];
}

ಸ್ಥಿರ vk_pipeline ggml_vk_get_dequantize_mul_mat_vec(ggml_backend_vk_context * ctx, ggml_type a_type, ggml_type b_type, uint32_t ಅಂಕೆ_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ, uint32_t m, uint32_t k) {
    VK_LOG_DEBUG("ggml_vk_get_dequantize_mul_mat_vec()");
    GGML_ASSERT(b_type == GGML_TYPE_F32 || b_type == GGML_TYPE_F16 || b_type == GGML_TYPE_Q8_1);
    GGML_ASSERT(ಅಂಕೆ_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ >= 1 && ಅಂಕೆ_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ <= mul_mat_vec_max_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ);

    ಆದರೆ (b_type == GGML_TYPE_Q8_1) {
        ಆಯ್ಕೆ (a_type) {
            ಸ್ಥಿತಿ GGML_TYPE_Q2_0:
            ಸ್ಥಿತಿ GGML_TYPE_Q4_0:
            ಸ್ಥಿತಿ GGML_TYPE_Q4_1:
            ಸ್ಥಿತಿ GGML_TYPE_Q5_0:
            ಸ್ಥಿತಿ GGML_TYPE_Q5_1:
            ಸ್ಥಿತಿ GGML_TYPE_Q8_0:
            ಸ್ಥಿತಿ GGML_TYPE_MXFP4:
            ಸ್ಥಿತಿ GGML_TYPE_Q2_K:
            ಸ್ಥಿತಿ GGML_TYPE_Q3_K:
            ಸ್ಥಿತಿ GGML_TYPE_Q4_K:
            ಸ್ಥಿತಿ GGML_TYPE_Q5_K:
            ಸ್ಥಿತಿ GGML_TYPE_Q6_K:
            ಸ್ಥಿತಿ GGML_TYPE_IQ1_S:
            ಸ್ಥಿತಿ GGML_TYPE_IQ1_M:
            ಸ್ಥಿತಿ GGML_TYPE_IQ4_XS:
                ನಿಲ್ಲಿಸು;
            ಪೂರ್ವನಿಯೋಜಿತ:
                ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
        }
    }

    ಆಯ್ಕೆ (a_type) {
        ಸ್ಥಿತಿ GGML_TYPE_F32:
        ಸ್ಥಿತಿ GGML_TYPE_F16:
        ಸ್ಥಿತಿ GGML_TYPE_BF16:
        ಸ್ಥಿತಿ GGML_TYPE_Q1_0:
        ಸ್ಥಿತಿ GGML_TYPE_Q2_0:
        ಸ್ಥಿತಿ GGML_TYPE_Q4_0:
        ಸ್ಥಿತಿ GGML_TYPE_Q4_1:
        ಸ್ಥಿತಿ GGML_TYPE_Q5_0:
        ಸ್ಥಿತಿ GGML_TYPE_Q5_1:
        ಸ್ಥಿತಿ GGML_TYPE_Q8_0:
        ಸ್ಥಿತಿ GGML_TYPE_Q2_K:
        ಸ್ಥಿತಿ GGML_TYPE_Q3_K:
        ಸ್ಥಿತಿ GGML_TYPE_Q4_K:
        ಸ್ಥಿತಿ GGML_TYPE_Q5_K:
        ಸ್ಥಿತಿ GGML_TYPE_Q6_K:
        ಸ್ಥಿತಿ GGML_TYPE_IQ1_S:
        ಸ್ಥಿತಿ GGML_TYPE_IQ1_M:
        ಸ್ಥಿತಿ GGML_TYPE_IQ2_XXS:
        ಸ್ಥಿತಿ GGML_TYPE_IQ2_XS:
        ಸ್ಥಿತಿ GGML_TYPE_IQ2_S:
        ಸ್ಥಿತಿ GGML_TYPE_IQ3_XXS:
        ಸ್ಥಿತಿ GGML_TYPE_IQ3_S:
        ಸ್ಥಿತಿ GGML_TYPE_IQ4_XS:
        ಸ್ಥಿತಿ GGML_TYPE_IQ4_NL:
        ಸ್ಥಿತಿ GGML_TYPE_MXFP4:
        ಸ್ಥಿತಿ GGML_TYPE_NVFP4:
        ಸ್ಥಿತಿ GGML_TYPE_TQ1_0:
        ಸ್ಥಿತಿ GGML_TYPE_TQ2_0:
            ನಿಲ್ಲಿಸು;
        ಪೂರ್ವನಿಯೋಜಿತ:
            ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
    }

    // heuristic to choose workgroup size
    uint32_t dmmv_wg = DMMV_WG_ಗಾತ್ರ_SUBGROUP;
    ಆದರೆ ((ctx->device->vendor_id == VK_VENDOR_ID_NVIDIA && ctx->device->architecture != vk_device_architecture::NVIDIA_PRE_TURING) || ctx->device->vendor_id == VK_VENDOR_ID_INTEL) {
        // Prefer larger workgroups when M is small, to spread the work out more
        // and keep more SMs busy.
        // q6_k seems to prefer small workgroup size even for "medium" values of M.
        ಆದರೆ (a_type == GGML_TYPE_Q6_K) {
            ಆದರೆ (m < 4096 && k >= 1024) {
                dmmv_wg = DMMV_WG_ಗಾತ್ರ_LARGE;
            }
        } ಇಲ್ಲದಿದ್ದರೆ {
            ಆದರೆ (m <= 8192 && k >= 1024) {
                dmmv_wg = DMMV_WG_ಗಾತ್ರ_LARGE;
            }
        }
    }

    ಆದರೆ (b_type == GGML_TYPE_Q8_1) {
        ಆದರೆ (ctx->device->vendor_id == VK_VENDOR_ID_INTEL) {
            dmmv_wg = DMMV_WG_ಗಾತ್ರ_SUBGROUP;
        }
        ಹಿಂತಿರುಗಿ ctx->device->pipeline_dequant_mul_mat_vec_q8_1_f32[dmmv_wg][a_type][ಅಂಕೆ_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ-1];
    }

    ಹಿಂತಿರುಗಿ b_type == GGML_TYPE_F32 ? ctx->device->pipeline_dequant_mul_mat_vec_f32_f32[dmmv_wg][a_type][ಅಂಕೆ_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ-1] : ctx->device->pipeline_dequant_mul_mat_vec_f16_f32[dmmv_wg][a_type][ಅಂಕೆ_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ-1];
}

ಸ್ಥಿರ vk_pipeline ggml_vk_get_dequantize_mul_mat_vec_id(ggml_backend_vk_context * ctx, ggml_type a_type, ggml_type b_type, uint32_t m, uint32_t k) {
    VK_LOG_DEBUG("ggml_vk_get_dequantize_mul_mat_vec_id()");
    GGML_ASSERT(b_type == GGML_TYPE_F32 || b_type == GGML_TYPE_Q8_1);

    ಆದರೆ (b_type == GGML_TYPE_Q8_1) {
        ಆಯ್ಕೆ (a_type) {
            ಸ್ಥಿತಿ GGML_TYPE_Q2_0:
            ಸ್ಥಿತಿ GGML_TYPE_Q4_0:
            ಸ್ಥಿತಿ GGML_TYPE_Q4_1:
            ಸ್ಥಿತಿ GGML_TYPE_Q5_0:
            ಸ್ಥಿತಿ GGML_TYPE_Q5_1:
            ಸ್ಥಿತಿ GGML_TYPE_Q8_0:
            ಸ್ಥಿತಿ GGML_TYPE_MXFP4:
            ಸ್ಥಿತಿ GGML_TYPE_Q2_K:
            ಸ್ಥಿತಿ GGML_TYPE_Q3_K:
            ಸ್ಥಿತಿ GGML_TYPE_Q4_K:
            ಸ್ಥಿತಿ GGML_TYPE_Q5_K:
            ಸ್ಥಿತಿ GGML_TYPE_Q6_K:
            ಸ್ಥಿತಿ GGML_TYPE_IQ1_S:
            ಸ್ಥಿತಿ GGML_TYPE_IQ1_M:
            ಸ್ಥಿತಿ GGML_TYPE_IQ4_XS:
                ನಿಲ್ಲಿಸು;
            ಪೂರ್ವನಿಯೋಜಿತ:
                ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
        }
    }

    ಆಯ್ಕೆ (a_type) {
        ಸ್ಥಿತಿ GGML_TYPE_F32:
        ಸ್ಥಿತಿ GGML_TYPE_F16:
        ಸ್ಥಿತಿ GGML_TYPE_BF16:
        ಸ್ಥಿತಿ GGML_TYPE_Q1_0:
        ಸ್ಥಿತಿ GGML_TYPE_Q2_0:
        ಸ್ಥಿತಿ GGML_TYPE_Q4_0:
        ಸ್ಥಿತಿ GGML_TYPE_Q4_1:
        ಸ್ಥಿತಿ GGML_TYPE_Q5_0:
        ಸ್ಥಿತಿ GGML_TYPE_Q5_1:
        ಸ್ಥಿತಿ GGML_TYPE_Q8_0:
        ಸ್ಥಿತಿ GGML_TYPE_Q2_K:
        ಸ್ಥಿತಿ GGML_TYPE_Q3_K:
        ಸ್ಥಿತಿ GGML_TYPE_Q4_K:
        ಸ್ಥಿತಿ GGML_TYPE_Q5_K:
        ಸ್ಥಿತಿ GGML_TYPE_Q6_K:
        ಸ್ಥಿತಿ GGML_TYPE_IQ1_S:
        ಸ್ಥಿತಿ GGML_TYPE_IQ1_M:
        ಸ್ಥಿತಿ GGML_TYPE_IQ2_XXS:
        ಸ್ಥಿತಿ GGML_TYPE_IQ2_XS:
        ಸ್ಥಿತಿ GGML_TYPE_IQ2_S:
        ಸ್ಥಿತಿ GGML_TYPE_IQ3_XXS:
        ಸ್ಥಿತಿ GGML_TYPE_IQ3_S:
        ಸ್ಥಿತಿ GGML_TYPE_IQ4_XS:
        ಸ್ಥಿತಿ GGML_TYPE_IQ4_NL:
        ಸ್ಥಿತಿ GGML_TYPE_MXFP4:
        ಸ್ಥಿತಿ GGML_TYPE_NVFP4:
        ಸ್ಥಿತಿ GGML_TYPE_TQ1_0:
        ಸ್ಥಿತಿ GGML_TYPE_TQ2_0:
            ನಿಲ್ಲಿಸು;
        ಪೂರ್ವನಿಯೋಜಿತ:
            ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
    }

    // heuristic to choose workgroup size
    uint32_t dmmv_wg = DMMV_WG_ಗಾತ್ರ_SUBGROUP;
    ಆದರೆ ((ctx->device->vendor_id == VK_VENDOR_ID_NVIDIA && ctx->device->architecture != vk_device_architecture::NVIDIA_PRE_TURING) || ctx->device->vendor_id == VK_VENDOR_ID_INTEL) {
        // Prefer larger workgroups when M is small, to spread the work out more
        // and keep more SMs busy.
        // q6_k seems to prefer small workgroup size even for "medium" values of M.
        ಆದರೆ (a_type == GGML_TYPE_Q6_K) {
            ಆದರೆ (m < 4096 && k >= 1024) {
                dmmv_wg = DMMV_WG_ಗಾತ್ರ_LARGE;
            }
        } ಇಲ್ಲದಿದ್ದರೆ {
            ಆದರೆ (m <= 8192 && k >= 1024) {
                dmmv_wg = DMMV_WG_ಗಾತ್ರ_LARGE;
            }
        }
    }

    ಆದರೆ (b_type == GGML_TYPE_Q8_1) {
        ಆದರೆ (ctx->device->vendor_id == VK_VENDOR_ID_INTEL) {
            dmmv_wg = DMMV_WG_ಗಾತ್ರ_SUBGROUP;
        }
        ಹಿಂತಿರುಗಿ ctx->device->pipeline_dequant_mul_mat_vec_id_q8_1_f32[dmmv_wg][a_type];
    }

    ಹಿಂತಿರುಗಿ ctx->device->pipeline_dequant_mul_mat_vec_id_f32[dmmv_wg][a_type];
}

vk_subbuffer ggml_vk_tensor_subbuffer(
    ಸ್ಥಿರಾಂಕ ggml_backend_vk_context * ctx, ಸ್ಥಿರಾಂಕ ggml_tensor * tensor, ತಾರ್ಕಿಕ allow_misalign) {

    vk_ಬಫರ್ ಬಫರ್ = ಶೂನ್ಯಸೂಚಕ;
    size_t ಸ್ಥಾನಾಂತರ = 0;
    ಆದರೆ (ctx->device->uma) {
        ggml_vk_host_get(ctx->device, tensor->ದತ್ತಾಂಶ, ಬಫರ್, ಸ್ಥಾನಾಂತರ);
    }
    ಆದರೆ (!ಬಫರ್) {
        ಸ್ವಯಂಚಾಲಿತ buf_ctx = (ggml_backend_vk_ಬಫರ್_context *)tensor->ಬಫರ್->context;
        ಬಫರ್ = buf_ctx->dev_ಬಫರ್;
        ಸ್ಥಾನಾಂತರ = vk_tensor_ಸ್ಥಾನಾಂತರ(tensor) + tensor->view_offs;
    }
    GGML_ASSERT(ಬಫರ್ != ಶೂನ್ಯಸೂಚಕ);

    size_t size = ggml_nbytes(tensor);

    ಸ್ಥಿರಾಂಕ size_t descriptor_ಸ್ಥಾನಾಂತರ = ggml_vk_descriptor_ಸ್ಥಾನಾಂತರ(
        ಸ್ಥಾನಾಂತರ, ctx->device->properties.limits.min_Storage_ಬಫರ್_ಸ್ಥಾನಾಂತರ_Alignment, ggml_type_ಗಾತ್ರ(tensor->type));
    ಸ್ಥಿರಾಂಕ size_t misalign_ಬೈಟುಗಳು = ಸ್ಥಾನಾಂತರ - descriptor_ಸ್ಥಾನಾಂತರ;
    // The shader must support misaligned offsets when indexing into the buffer
    GGML_ASSERT(allow_misalign || misalign_ಬೈಟುಗಳು == 0);
    ಸ್ಥಾನಾಂತರ = descriptor_ಸ್ಥಾನಾಂತರ;
    size += misalign_ಬೈಟುಗಳು;

    ಹಿಂತಿರುಗಿ vk_subbuffer{ಬಫರ್, ಸ್ಥಾನಾಂತರ, size};
}

ಸ್ಥಿರ vk_command_ಬಫರ್* ggml_vk_get_or_create_cmd_ಬಫರ್(vk_device& device, vk_command_pool& pool) {
    ಗಾಗಿ (ಸ್ವಯಂಚಾಲಿತ& cmd_ಬಫರ್ : pool.cmd_buffers) {
        ಆದರೆ (!cmd_ಬಫರ್.ಒಳ_use) {
            cmd_ಬಫರ್.use_counter++;
            cmd_ಬಫರ್.ಒಳ_use = ನಿಜ;
            ಹಿಂತಿರುಗಿ &cmd_ಬಫರ್;
        }
    }
    ಹಿಂತಿರುಗಿ ggml_vk_create_cmd_ಬಫರ್(device, pool);
}

ಸ್ಥಿರ vk_submission ggml_vk_begin_submission(vk_device& device, vk_command_pool& p, ತಾರ್ಕಿಕ one_ಸಮಯ = ನಿಜ) {
    vk_submission s;
    s.ಬಫರ್ = ggml_vk_get_or_create_cmd_ಬಫರ್(device, p);
    ಆದರೆ (one_ಸಮಯ) {
        s.ಬಫರ್->buf.begin({ vk::Command_ಬಫರ್_Usage_Flag_ಬಿಟ್ಟುಗಳು::e_One_ಸಮಯ_Submit });
    } ಇಲ್ಲದಿದ್ದರೆ {
        s.ಬಫರ್->buf.begin({ vk::Command_ಬಫರ್_Usage_Flags{} });
    }

    ಹಿಂತಿರುಗಿ s;
}

ಶೂನ್ಯ ggml_vk_cmd_label_begin(vk::Command_ಬಫರ್ buf, ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * name) {
    vk::DebugUtilsLabelEXT label = {};
    label.pLabelName = name;
    label.color = std::array<ದಶಮಾಂಶ, 4>{1.0f, 1.0f, 1.0f, 1.0f};
    vk_instance.pfn_vkCmdBeginDebugUtilsLabelEXT(buf, reinterpret_cast<VkDebugUtilsLabelEXT *>(&label));
}

ಶೂನ್ಯ ggml_vk_ctx_ಕೊನೆ(vk_context& ctx) {
    VK_LOG_DEBUG("ggml_vk_ctx_end(" << ctx << ", " << ctx->seqs.size() << ")");
    ಆದರೆ (ctx->s == ಶೂನ್ಯಸೂಚಕ) {
        ಹಿಂತಿರುಗಿ;
    }

    // close open labels so this buffer is balanced; reopened in ggml_vk_ctx_begin
    ಆದರೆ (vk_instance.debug_utils_support) {
        ಗಾಗಿ (size_t i = 0; i < ctx->debug_labels.size(); i++) {
            vk_instance.pfn_vk_Cmd_ಕೊನೆ_Debug_Utils_Label_EXT(ctx->s->ಬಫರ್->buf);
        }
        // the enclosing per-command-buffer region
        vk_instance.pfn_vk_Cmd_ಕೊನೆ_Debug_Utils_Label_EXT(ctx->s->ಬಫರ್->buf);
    }

    ctx->s->ಬಫರ್->buf.end();
    ctx->s = ಶೂನ್ಯಸೂಚಕ;
}

ಶೂನ್ಯ ggml_vk_ctx_begin(vk_device& device, vk_context& subctx) {
    VK_LOG_DEBUG("ggml_vk_ctx_begin(" << device->name << ")");
    ಆದರೆ (subctx->s != ಶೂನ್ಯಸೂಚಕ) {
        ggml_vk_ctx_ಕೊನೆ(subctx);
    }

    subctx->seqs.push_back({ ggml_vk_begin_submission(device, *subctx->p) });
    subctx->s = subctx->seqs[subctx->seqs.size() - 1].ದತ್ತಾಂಶ();

    ಆದರೆ (vk_instance.debug_utils_support) {
        // outermost region, one per command buffer, so the gaps between submits stand out
        ಸ್ಥಿರಾಂಕ std::string name = "submit " + std::to_string(device->debug_cmdbuf_idx++);
        ggml_vk_cmd_label_begin(subctx->s->ಬಫರ್->buf, name.c_str());

        // reopen labels left open when the previous command buffer was submitted
        ಗಾಗಿ (ಸ್ಥಿರಾಂಕ std::string & label : subctx->debug_labels) {
            ggml_vk_cmd_label_begin(subctx->s->ಬಫರ್->buf, label.c_str());
        }
    }
}

vk_context ggml_vk_get_compute_ctx(ggml_backend_vk_context * ctx) {
    vk_context ಫಲಿತಾಂಶ;
    ಆದರೆ (!ctx->compute_ctx.expired()) {
        ಫಲಿತಾಂಶ = ctx->compute_ctx.ಬೀಗ();
    } ಇಲ್ಲದಿದ್ದರೆ {
        ಫಲಿತಾಂಶ = ggml_vk_create_context(ctx, ctx->compute_cmd_pool);

        ctx->compute_ctx = ಫಲಿತಾಂಶ;
        ggml_vk_ctx_begin(ctx->device, ಫಲಿತಾಂಶ);
    }

    ಆದರೆ (ctx->device->async_use_transfer_ಸರತಿ && ctx->transfer_semaphore_ಕೊನೆಯ_submitted < ctx->transfer_semaphore.value) {
        ಫಲಿತಾಂಶ->s->wait_semaphores.push_back(ctx->transfer_semaphore);
        ctx->transfer_semaphore_ಕೊನೆಯ_submitted = ctx->transfer_semaphore.value;
    }

    ಹಿಂತಿರುಗಿ ಫಲಿತಾಂಶ;
}

vk_context ggml_vk_get_transfer_ctx(ggml_backend_vk_context * ctx) {
    vk_context ಫಲಿತಾಂಶ;
    ಆದರೆ (!ctx->transfer_ctx.expired()) {
        ಫಲಿತಾಂಶ = ctx->transfer_ctx.ಬೀಗ();
    } ಇಲ್ಲದಿದ್ದರೆ {
        ಫಲಿತಾಂಶ = ggml_vk_create_context(ctx, ctx->transfer_cmd_pool);

        ctx->transfer_ctx = ಫಲಿತಾಂಶ;
        ggml_vk_ctx_begin(ctx->device, ಫಲಿತಾಂಶ);
    }

    ಹಿಂತಿರುಗಿ ಫಲಿತಾಂಶ;
}

ತಾರ್ಕಿಕ ggml_vk_submit_transfer_ctx(ggml_backend_vk_context * ctx) {
    ಆದರೆ (!ctx->device->async_use_transfer_ಸರತಿ || ctx->transfer_ctx.expired()) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    vk_context cpy_ctx = ctx->transfer_ctx.ಬೀಗ();
    ggml_vk_ctx_ಕೊನೆ(cpy_ctx);

    ಗಾಗಿ (ಸ್ವಯಂಚಾಲಿತ& cpy : cpy_ctx->ಒಳ_memcpys) {
        ಸ್ಮರಣೆ_ನಕಲು(cpy.dst, cpy.src, cpy.n);
    }

    ctx->transfer_semaphore.value++;
    cpy_ctx->seqs.back().back().signal_semaphores.push_back(ctx->transfer_semaphore);

    ggml_vk_submit(cpy_ctx, {});
    ctx->transfer_ctx.reset();
    ಹಿಂತಿರುಗಿ ನಿಜ;
}

size_t ggml_vk_align_ಗಾತ್ರ(size_t ಅಗಲ, size_t align) {
    VK_LOG_DEBUG("ggml_vk_align_size(" << ಅಗಲ << ", " << align << ")");
    ಹಿಂತಿರುಗಿ CEIL_DIV(ಅಗಲ, align) * align;
}

ಶೂನ್ಯ deferred_memcpy(ಶೂನ್ಯ * dst, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * src, size_t size, std::vector<vk_staging_memcpy>* memcpys) {
    ಆದರೆ (memcpys == ಶೂನ್ಯಸೂಚಕ) {
        ಸ್ಮರಣೆ_ನಕಲು(dst, src, size);
    } ಇಲ್ಲದಿದ್ದರೆ {
        memcpys->emplace_back(dst, src, size);
    }
}

ಶೂನ್ಯ deferred_memset(ಶೂನ್ಯ * dst, uint32_t val, size_t size, std::vector<vk_staging_memset>* memsets) {
    ಆದರೆ (memsets == ಶೂನ್ಯಸೂಚಕ) {
        memset(dst, val, size);
    } ಇಲ್ಲದಿದ್ದರೆ {
        memsets->emplace_back(dst, val, size);
    }
}

ಸ್ಥಿರ uint32_t ggml_vk_guess_ವಿಭಜಿಸು_k(ggml_backend_vk_context * ctx, uint32_t m, uint32_t n, uint32_t k, ತಾರ್ಕಿಕ disable_ವಿಭಜಿಸು_k, ಸ್ಥಿರಾಂಕ vk_pipeline& pipeline) {
    VK_LOG_DEBUG("ggml_vk_guess_split_k(" << m << ", " << n << ", " << k << ", " << disable_ವಿಭಜಿಸು_k << ")");

    ಆದರೆ (disable_ವಿಭಜಿಸು_k) {
        ಹಿಂತಿರುಗಿ 1;
    }

    uint32_t ವಿಭಜಿಸು_k = 1;
    ಆದರೆ (ctx->device->shader_core_ಎಣಿಕೆ != 0 && n >= pipeline->wg_denoms[1]) {
        // If k is 'large' and the SMs will fill less than halfway, use split_k.
        uint32_t m_tiles = CEIL_DIV(m, pipeline->wg_denoms[0]);
        uint32_t n_tiles = CEIL_DIV(n, pipeline->wg_denoms[1]);

        ಆದರೆ (k >= 2048) {
            ಆದರೆ (m_tiles * n_tiles <= ctx->device->shader_core_ಎಣಿಕೆ / 2) {
                ವಿಭಜಿಸು_k = ctx->device->shader_core_ಎಣಿಕೆ / (m_tiles * n_tiles);
            } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (m_tiles * n_tiles <= ctx->device->shader_core_ಎಣಿಕೆ * 2 / 3) {
                ವಿಭಜಿಸು_k = 3;
            }
            // Cap the split at 8x. Unless k is huge this is a lot of overhead.
            ವಿಭಜಿಸು_k = std::min(ವಿಭಜಿಸು_k, 8u);

            // ggml_vk_matmul will align the splits to be a multiple of 256.
            // If this rounded up size would cause the last split to be empty,
            // then reduce the split count.
            ಆಗ (ನಿಜ) {
                ಆದರೆ (ವಿಭಜಿಸು_k == 1) {
                    ನಿಲ್ಲಿಸು;
                }
                uint32_t k_ವಿಭಜಿಸು = CEIL_DIV(k, ವಿಭಜಿಸು_k);
                k_ವಿಭಜಿಸು = ROUNDUP_POW2(k_ವಿಭಜಿಸು, 256);
                ಆದರೆ (k_ವಿಭಜಿಸು * (ವಿಭಜಿಸು_k - 1) < k) {
                    ನಿಲ್ಲಿಸು;
                }
                ವಿಭಜಿಸು_k--;
            }
        }
    }

    ಹಿಂತಿರುಗಿ ವಿಭಜಿಸು_k;
}

ಶೂನ್ಯ ggml_vk_matmul(
        ggml_backend_vk_context * ctx, vk_context& subctx, vk_pipeline& pipeline,
        vk_subbuffer&& a, vk_subbuffer&& b, vk_subbuffer&& d, vk_subbuffer&& ವಿಭಜಿಸು_k_ಬಫರ್,
        uint32_t m, uint32_t n, uint32_t k, uint32_t stride_a, uint32_t stride_b, uint32_t stride_d,
        uint32_t batch_stride_a, uint32_t batch_stride_b, uint32_t batch_stride_d,
        uint32_t ವಿಭಜಿಸು_k, uint32_t batch, uint32_t ne02, uint32_t ne12, uint32_t broadcast2, uint32_t broadcast3,
        uint32_t padded_n, uint64_t b_ಲಂಬಸಾಲು_ಬೈಟುಗಳು, uint64_t d_ಲಂಬಸಾಲು_ಬೈಟುಗಳು) {
        VK_LOG_DEBUG("ggml_vk_matmul(a: (" << a.ಬಫರ್->ಬಫರ್ << ", " << a.ಸ್ಥಾನಾಂತರ << ", " << a.size << "), b: (" << b.ಬಫರ್->ಬಫರ್ << ", " << b.ಸ್ಥಾನಾಂತರ << ", " << b.size << "), d: (" << d.ಬಫರ್->ಬಫರ್ << ", " << d.ಸ್ಥಾನಾಂತರ << ", " << d.size << "), split_k: (" << (ವಿಭಜಿಸು_k_ಬಫರ್.ಬಫರ್ != ಶೂನ್ಯಸೂಚಕ ? ವಿಭಜಿಸು_k_ಬಫರ್.ಬಫರ್->ಬಫರ್ : VK_NULL_HANDLE) << ", " << ವಿಭಜಿಸು_k_ಬಫರ್.ಸ್ಥಾನಾಂತರ << ", " << ವಿಭಜಿಸು_k_ಬಫರ್.size << "), m: " << m << ", n: " << n << ", k: " << k << ", stride_a: " << stride_a << ", stride_b: " << stride_b << ", stride_d: " << stride_d << ", batch_stride_a: " << batch_stride_a << ", batch_stride_b: " << batch_stride_b << ", batch_stride_d: " << batch_stride_d << ", split_k: " << ವಿಭಜಿಸು_k << ", batch: " << batch << ", ne02: " << ne02 << ", ne12: " << ne12 << ", broadcast2: " << broadcast2 << ", broadcast3: " << broadcast3 << ", padded_n: " << padded_n << ")");
    ಆದರೆ (ವಿಭಜಿಸು_k == 1) {
        // a small tile can still produce more workgroups than the device allows on the n axis:
        // split the dispatch into n-chunks, shifting the B and D column windows
        ಸ್ಥಿರಾಂಕ uint64_t max_n = (uint64_t)ctx->device->properties.limits.max_Compute_Work_ಗುಂಪು_ಎಣಿಕೆ[1] * pipeline->wg_denoms[1];
        ಸ್ಥಿರಾಂಕ uint64_t n_chunks = CEIL_DIV(n, max_n);
        ggml_pipeline_request_descriptor_sets(ctx, pipeline, n_chunks * CEIL_DIV(batch, ctx->device->properties.limits.max_Compute_Work_ಗುಂಪು_ಎಣಿಕೆ[2]));

        ಗಾಗಿ (uint64_t n_ತಳ = 0; n_ತಳ < n; n_ತಳ += max_n) {
            ಸ್ಥಿರಾಂಕ uint32_t n_chunk = (uint32_t) std::min(n - n_ತಳ, max_n);
            GGML_ASSERT(n_ತಳ * b_ಲಂಬಸಾಲು_ಬೈಟುಗಳು < b.size && n_ತಳ * d_ಲಂಬಸಾಲು_ಬೈಟುಗಳು < d.size);
            GGML_ASSERT(b.size != VK_ಸಂಪೂರ್ಣ_ಗಾತ್ರ && d.size != VK_ಸಂಪೂರ್ಣ_ಗಾತ್ರ);
            vk_subbuffer b_chunk = b;
            vk_subbuffer d_chunk = d;
            b_chunk.ಸ್ಥಾನಾಂತರ += n_ತಳ * b_ಲಂಬಸಾಲು_ಬೈಟುಗಳು;
            b_chunk.size   -= n_ತಳ * b_ಲಂಬಸಾಲು_ಬೈಟುಗಳು;
            d_chunk.ಸ್ಥಾನಾಂತರ += n_ತಳ * d_ಲಂಬಸಾಲು_ಬೈಟುಗಳು;
            d_chunk.size   -= n_ತಳ * d_ಲಂಬಸಾಲು_ಬೈಟುಗಳು;

            uint32_t ತಳ_work_ಗುಂಪು_z = 0;
            ಆಗ (ತಳ_work_ಗುಂಪು_z < batch) {
                uint32_t groups_z = std::min(batch - ತಳ_work_ಗುಂಪು_z, ctx->device->properties.limits.max_Compute_Work_ಗುಂಪು_ಎಣಿಕೆ[2]);

                ಸ್ಥಿರಾಂಕ vk_mat_mat_ತಳ್ಳು_constants pc = { m, n_chunk, k, stride_a, stride_b, stride_d, batch_stride_a, batch_stride_b, batch_stride_d, ತಳ_work_ಗುಂಪು_z, batch, k, ne02, ne12, broadcast2, broadcast3, uint32_t(padded_n - n_ತಳ) };
                ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { a, b_chunk, d_chunk }, pc, { m, n_chunk, groups_z });
                ತಳ_work_ಗುಂಪು_z += groups_z;
            }
        }
        ಹಿಂತಿರುಗಿ;
    }

    ಆದರೆ (ctx->prealloc_ವಿಭಜಿಸು_k_need_sync) {
        ggml_vk_sync_buffers(ctx, subctx);
    }

    GGML_ASSERT(batch_stride_d == m * n);

    // Round the split size up to a multiple of 256 (k-quant alignment)
    uint32_t k_ವಿಭಜಿಸು = CEIL_DIV(k, ವಿಭಜಿಸು_k);
    k_ವಿಭಜಿಸು = ROUNDUP_POW2(k_ವಿಭಜಿಸು, 256);

    ggml_pipeline_request_descriptor_sets(ctx, pipeline, CEIL_DIV(batch, ctx->device->properties.limits.max_Compute_Work_ಗುಂಪು_ಎಣಿಕೆ[2]));

    uint32_t ತಳ_work_ಗುಂಪು_z = 0;
    ಆಗ (ತಳ_work_ಗುಂಪು_z < batch) {
        uint32_t groups_z = std::min(batch - ತಳ_work_ಗುಂಪು_z, ctx->device->properties.limits.max_Compute_Work_ಗುಂಪು_ಎಣಿಕೆ[2]);

        ಸ್ಥಿರಾಂಕ vk_mat_mat_ತಳ್ಳು_constants pc1 = { m, n, k, stride_a, stride_b, stride_d, batch_stride_a, batch_stride_b, batch_stride_d, ತಳ_work_ಗುಂಪು_z, batch, k_ವಿಭಜಿಸು, ne02, ne12, broadcast2, broadcast3, padded_n };
        // Make sure enough workgroups get assigned for split k to work
        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { a, b, ವಿಭಜಿಸು_k_ಬಫರ್ }, pc1, { (CEIL_DIV(m, pipeline->wg_denoms[0]) * pipeline->wg_denoms[0]) * ವಿಭಜಿಸು_k, n, groups_z });
        ತಳ_work_ಗುಂಪು_z += groups_z;
    }
    ggml_vk_sync_buffers(ctx, subctx);
    ಸ್ಥಿರಾಂಕ std::array<uint32_t, 2> pc2 = { (uint32_t)(m * n * batch), ವಿಭಜಿಸು_k };
    ggml_vk_dispatch_pipeline(ctx, subctx, ctx->device->pipeline_matmul_ವಿಭಜಿಸು_k_reduce, { ವಿಭಜಿಸು_k_ಬಫರ್, d }, pc2, { m * n * batch, 1, 1 });
    ctx->prealloc_ವಿಭಜಿಸು_k_need_sync = ನಿಜ;
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_vk_get_mul_mat_mat_f16acc(ggml_backend_vk_context * ctx, ggml_type src0_type, ggml_type src1_type, ggml_prec prec) {
    ಆದರೆ (src0_type == GGML_TYPE_F32 || src0_type == GGML_TYPE_BF16) ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    ಆದರೆ (src1_type == GGML_TYPE_Q8_1) ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    ಆದರೆ (src0_type == GGML_TYPE_F16) {
        ಹಿಂತಿರುಗಿ prec == GGML_PREC_DEFAULT && ctx->device->fp16 && !(ctx->device->coopmat_support && !ctx->device->coopmat_acc_f16_support);
    }
    // quant types
    ಆದರೆ (ctx->device->coopmat2) {
        ಹಿಂತಿರುಗಿ prec == GGML_PREC_DEFAULT;
    }
    ಆದರೆ (ctx->device->coopmat_support) {
        ಹಿಂತಿರುಗಿ ctx->device->fp16 && ctx->device->coopmat_acc_f16_support && prec == GGML_PREC_DEFAULT;
    }
    ಹಿಂತಿರುಗಿ ctx->device->fp16 && prec == GGML_PREC_DEFAULT;
}

ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ std::vector<vk_matmul_pipeline_pair>* ggml_vk_get_mul_mat_mat_pipeline_map(
        ggml_backend_vk_context * ctx, ggml_type src0_type, ggml_type src1_type, ggml_prec prec, ತಾರ್ಕಿಕ mul_mat_id = ಸುಳ್ಳು) {
    ತಾರ್ಕಿಕ f16acc = ggml_vk_get_mul_mat_mat_f16acc(ctx, src0_type, src1_type, prec);
    vk_matmul_pipeline_ಕೀಲಿ ಕೀಲಿ{src0_type, src1_type, mul_mat_id, f16acc};
    ಸ್ವಯಂಚಾಲಿತ it = ctx->device->pipeline_matmul.find(ಕೀಲಿ);
    ಆದರೆ (it == ctx->device->pipeline_matmul.end() || it->second.ಖಾಲಿಗಳು()) {
        // Try without f16acc
        ಆದರೆ (f16acc) {
            ಕೀಲಿ.f16acc = ಸುಳ್ಳು;
            it = ctx->device->pipeline_matmul.find(ಕೀಲಿ);
            ಆದರೆ (it != ctx->device->pipeline_matmul.end() && !it->second.ಖಾಲಿಗಳು()) ಹಿಂತಿರುಗಿ &it->second;
        }
        ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
    }
    ಹಿಂತಿರುಗಿ &it->second;
}

ಸ್ಥಿರ vk_pipeline ggml_vk_guess_matmul_pipeline_map(ggml_backend_vk_context * ctx,
        ಸ್ಥಿರಾಂಕ std::vector<vk_matmul_pipeline_pair>& configs,
        uint32_t m, uint32_t n, ತಾರ್ಕಿಕ aligned, ತಾರ್ಕಿಕ mul_mat_id) {
    ಸ್ವಯಂಚಾಲಿತ& selector = mul_mat_id ? ctx->device->matmul_id_tile_selector : ctx->device->matmul_tile_selector;
    uint32_t idx = selector(m, n, 0, ctx->device->shader_core_ಎಣಿಕೆ, configs);
    ಆದರೆ (idx >= configs.size()) idx = (uint32_t)configs.size() - 1;
    ಹಿಂತಿರುಗಿ (aligned && configs[idx].aligned) ? configs[idx].aligned : configs[idx].unaligned;
}

ಸ್ಥಿರ uint32_t ggml_vk_guess_matmul_pipeline_align_map(ggml_backend_vk_context * ctx,
        ಸ್ಥಿರಾಂಕ std::vector<vk_matmul_pipeline_pair>& configs,
        uint32_t m, uint32_t n, ತಾರ್ಕಿಕ mul_mat_id) {
    ಸ್ವಯಂಚಾಲಿತ& selector = mul_mat_id ? ctx->device->matmul_id_tile_selector : ctx->device->matmul_tile_selector;
    uint32_t idx = selector(m, n, 0, ctx->device->shader_core_ಎಣಿಕೆ, configs);
    ಆದರೆ (idx >= configs.size()) idx = (uint32_t)configs.size() - 1;
    ಹಿಂತಿರುಗಿ configs[idx].align;
}

ಸ್ಥಿರ uint64_t ggml_vk_mul_mat_id_max_tiles(uint64_t n_experts, uint64_t n_ಅಡ್ಡಸಾಲುಗಳು, uint32_t bn) {
    // Each nonempty expert can add at most one partial tile to the combined grid.
    ಹಿಂತಿರುಗಿ std::min(n_ಅಡ್ಡಸಾಲುಗಳು, CEIL_DIV(n_ಅಡ್ಡಸಾಲುಗಳು, bn) + std::min(n_experts, n_ಅಡ್ಡಸಾಲುಗಳು) - 1);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_vk_matmul_id(
        ggml_backend_vk_context * ctx, vk_context& subctx, vk_pipeline& pipeline,
        vk_subbuffer&& a, vk_subbuffer&& b, vk_subbuffer&& d, vk_subbuffer&& ids, ಸ್ಥಿರಾಂಕ vk_subbuffer & expert_ಎಣಿಕೆ_buf,
        uint32_t m, uint32_t n, uint32_t k, uint32_t stride_a, uint32_t stride_b, uint32_t stride_d,
        uint32_t batch_stride_a, uint32_t batch_stride_b, uint32_t batch_stride_d,
        uint32_t n_as, uint32_t nei0, uint32_t nei1, uint32_t nbi1, uint32_t ne11,
        uint32_t ಅಡ್ಡಸಾಲು_ids_ಸ್ಥಾನಾಂತರ, uint32_t max_tiles) {
    VK_LOG_DEBUG("ggml_vk_matmul_id(a: (" << a.ಬಫರ್->ಬಫರ್ << ", " << a.ಸ್ಥಾನಾಂತರ << ", " << a.size << "), b: (" << b.ಬಫರ್->ಬಫರ್ << ", " << b.ಸ್ಥಾನಾಂತರ << ", " << b.size << "), d: (" << d.ಬಫರ್->ಬಫರ್ << ", " << d.ಸ್ಥಾನಾಂತರ << ", " << d.size << "), ids: (" << ids.ಬಫರ್->ಬಫರ್ << ", " << ids.ಸ್ಥಾನಾಂತರ << ", " << ids.size << "), expert_count: (" << expert_ಎಣಿಕೆ_buf.ಬಫರ್->ಬಫರ್ << ", " << expert_ಎಣಿಕೆ_buf.ಸ್ಥಾನಾಂತರ << ", " << expert_ಎಣಿಕೆ_buf.size << "), " <<
        "m: " << m << ", n: " << n << ", k: " << k << ", stride_a: " << stride_a << ", stride_b: " << stride_b << ", stride_d: " << stride_d << ", " <<
        "batch_stride_a: " << batch_stride_a << ", batch_stride_b: " << batch_stride_b << ", batch_stride_d: " << batch_stride_d << ", " <<
        "n_as: " << n_as << ", nei0: " << nei0 << ", nei1: " << nei1 << ", nbi1: " << nbi1 << ", ne11: " << ne11 << ")");
    ಸ್ಥಿರಾಂಕ vk_mat_mat_id_ತಳ್ಳು_constants pc = { m, n, k, stride_a, stride_b, stride_d, batch_stride_a, batch_stride_b, batch_stride_d,
                                              nei0, nei1, nbi1, ne11, n_as, ಅಡ್ಡಸಾಲು_ids_ಸ್ಥಾನಾಂತರ };
    ಸ್ಥಿರಾಂಕ uint32_t tiles_y = std::min(max_tiles, ctx->device->properties.limits.max_Compute_Work_ಗುಂಪು_ಎಣಿಕೆ[1]);
    ಸ್ಥಿರಾಂಕ std::array<uint32_t, 3> elements = { m, tiles_y * pipeline->wg_denoms[1], uint32_t(CEIL_DIV(max_tiles, tiles_y)) };
    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { a, b, d, ids, expert_ಎಣಿಕೆ_buf }, pc, elements);
}

ತಾರ್ಕಿಕ ggml_vk_dim01_contiguous(ಸ್ಥಿರಾಂಕ ggml_tensor * tensor) {
    ಹಿಂತಿರುಗಿ
        tensor->nb[0] == ggml_type_ಗಾತ್ರ(tensor->type) &&
        tensor->nb[1] == (tensor->nb[0]*tensor->ne[0])/ggml_blck_ಗಾತ್ರ(tensor->type) &&
        (tensor->ne[3] == 1 || tensor->nb[3] == tensor->nb[2]*tensor->ne[2]);
}

// Batch stride in elements of a tensor read in place.
ಸ್ಥಿರ uint32_t ggml_vk_batch_stride(ಸ್ಥಿರಾಂಕ ggml_tensor * tensor) {
    ಹಿಂತಿರುಗಿ (uint32_t)(tensor->nb[2] / ggml_type_ಗಾತ್ರ(tensor->type) * ggml_blck_ಗಾತ್ರ(tensor->type));
}

vk_pipeline ggml_vk_get_cpy_pipeline(ggml_backend_vk_context * ctx, ಸ್ಥಿರಾಂಕ ggml_tensor * src, ಸ್ಥಿರಾಂಕ ggml_tensor * dst, ggml_type ಗಮ್ಯ) {

    // Choose "contiguous copy" shader if src/dst are contiguous
    ತಾರ್ಕಿಕ contig = ggml_is_contiguous(src) && (!dst || ggml_is_contiguous(dst));

    // Use optimized "transpose" shader if src dim1 is the innermost dimension.
    ತಾರ್ಕಿಕ transpose = dst && src->nb[1] == ggml_type_ಗಾತ್ರ(ಗಮ್ಯ) && ggml_are_same_shape(dst, src);

    ಆದರೆ (transpose && src->type == ಗಮ್ಯ) {
        ಆದರೆ (ggml_type_ಗಾತ್ರ(ಗಮ್ಯ) == 4) {
            ಹಿಂತಿರುಗಿ ctx->device->pipeline_cpy_transpose_32;
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (ggml_type_ಗಾತ್ರ(ಗಮ್ಯ) == 2) {
            ಹಿಂತಿರುಗಿ ctx->device->pipeline_cpy_transpose_16;
        }
    }

    // Same, for a 0<->2 swap: src dim2 is the innermost dimension.
    ತಾರ್ಕಿಕ transpose02 = dst && !contig && src->nb[2] == ggml_type_ಗಾತ್ರ(ಗಮ್ಯ) &&
                       ggml_is_contiguous(dst) && ggml_are_same_shape(dst, src);

    ಆದರೆ (transpose02 && src->type == ಗಮ್ಯ) {
        ಆದರೆ (ggml_type_ಗಾತ್ರ(ಗಮ್ಯ) == 4) {
            ಹಿಂತಿರುಗಿ ctx->device->pipeline_cpy_transpose_02_32;
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (ggml_type_ಗಾತ್ರ(ಗಮ್ಯ) == 2) {
            ಹಿಂತಿರುಗಿ ctx->device->pipeline_cpy_transpose_02_16;
        }
    }

    ಆದರೆ (src->type == GGML_TYPE_F32 && ಗಮ್ಯ == GGML_TYPE_F16) {
        ಆದರೆ (contig) {
            ಹಿಂತಿರುಗಿ ctx->device->pipeline_contig_cpy_f32_f16;
        } ಇಲ್ಲದಿದ್ದರೆ {
            ಹಿಂತಿರುಗಿ ctx->device->pipeline_cpy_f32_f16;
        }
    }
    ಆದರೆ (src->type == GGML_TYPE_F16 && ಗಮ್ಯ == GGML_TYPE_F32) {
        ಆದರೆ (contig) {
            ಹಿಂತಿರುಗಿ ctx->device->pipeline_contig_cpy_f16_f32;
        } ಇಲ್ಲದಿದ್ದರೆ {
            ಹಿಂತಿರುಗಿ ctx->device->pipeline_cpy_f16_f32;
        }
    }
    ಆದರೆ (src->type == GGML_TYPE_F32 && ಗಮ್ಯ == GGML_TYPE_BF16) {
        ಆದರೆ (contig) {
            ಹಿಂತಿರುಗಿ ctx->device->pipeline_contig_cpy_f32_bf16;
        } ಇಲ್ಲದಿದ್ದರೆ {
            ಹಿಂತಿರುಗಿ ctx->device->pipeline_cpy_f32_bf16;
        }
    }
    ಆದರೆ (src->type == GGML_TYPE_BF16 && ಗಮ್ಯ == GGML_TYPE_F32) {
        ಆದರೆ (contig) {
            ಹಿಂತಿರುಗಿ ctx->device->pipeline_contig_cpy_bf16_f32;
        } ಇಲ್ಲದಿದ್ದರೆ {
            ಹಿಂತಿರುಗಿ ctx->device->pipeline_cpy_bf16_f32;
        }
    }
    ಆದರೆ (src->type == GGML_TYPE_BF16 && ಗಮ್ಯ == GGML_TYPE_F16) {
        ಆದರೆ (contig) {
            ಹಿಂತಿರುಗಿ ctx->device->pipeline_contig_cpy_bf16_f16;
        } ಇಲ್ಲದಿದ್ದರೆ {
            ಹಿಂತಿರುಗಿ ctx->device->pipeline_cpy_bf16_f16;
        }
    }
    ಆದರೆ (src->type == GGML_TYPE_F32 && ಗಮ್ಯ == GGML_TYPE_I32) {
        ಆದರೆ (contig) {
            ಹಿಂತಿರುಗಿ ctx->device->pipeline_contig_cpy_f32_i32;
        } ಇಲ್ಲದಿದ್ದರೆ {
            ಹಿಂತಿರುಗಿ ctx->device->pipeline_cpy_f32_i32;
        }
    }
    ಆದರೆ (src->type == GGML_TYPE_I32 && ಗಮ್ಯ == GGML_TYPE_F32) {
        ಆದರೆ (contig) {
            ಹಿಂತಿರುಗಿ ctx->device->pipeline_contig_cpy_i32_f32;
        } ಇಲ್ಲದಿದ್ದರೆ {
            ಹಿಂತಿರುಗಿ ctx->device->pipeline_cpy_i32_f32;
        }
    }
    ಆದರೆ (src->type == GGML_TYPE_F32) {
        ಆಯ್ಕೆ (ಗಮ್ಯ) {
        ಸ್ಥಿತಿ GGML_TYPE_Q1_0:
        ಸ್ಥಿತಿ GGML_TYPE_Q2_0:
        ಸ್ಥಿತಿ GGML_TYPE_Q4_0:
        ಸ್ಥಿತಿ GGML_TYPE_Q4_1:
        ಸ್ಥಿತಿ GGML_TYPE_Q5_0:
        ಸ್ಥಿತಿ GGML_TYPE_Q5_1:
        ಸ್ಥಿತಿ GGML_TYPE_Q8_0:
        ಸ್ಥಿತಿ GGML_TYPE_IQ4_NL:
            ಹಿಂತಿರುಗಿ ctx->device->pipeline_cpy_f32_quant[ಗಮ್ಯ];
        ಪೂರ್ವನಿಯೋಜಿತ:
            ನಿಲ್ಲಿಸು;
        }
    }

    ಆದರೆ (ಗಮ್ಯ == GGML_TYPE_F32) {
        ಆಯ್ಕೆ (src->type) {
        ಸ್ಥಿತಿ GGML_TYPE_Q1_0:
        ಸ್ಥಿತಿ GGML_TYPE_Q2_0:
        ಸ್ಥಿತಿ GGML_TYPE_Q4_0:
        ಸ್ಥಿತಿ GGML_TYPE_Q4_1:
        ಸ್ಥಿತಿ GGML_TYPE_Q5_0:
        ಸ್ಥಿತಿ GGML_TYPE_Q5_1:
        ಸ್ಥಿತಿ GGML_TYPE_Q8_0:
        ಸ್ಥಿತಿ GGML_TYPE_IQ4_NL:
            ಹಿಂತಿರುಗಿ ctx->device->pipeline_cpy_quant_f32[src->type];
        ಪೂರ್ವನಿಯೋಜಿತ:
            ನಿಲ್ಲಿಸು;
        }
    }

    ಆದರೆ (src->type == ಗಮ್ಯ) {
        // Copy two or four bytes at a time, depending on block size.
        // For quantized types, we scale by block size/type size. But
        // this path is also used for bf16->bf16 for example, where the
        // type size must be exactly 2 or 4.
        GGML_ASSERT(ggml_is_quantized(ಗಮ್ಯ) || ggml_type_ಗಾತ್ರ(src->type) == 2 || ggml_type_ಗಾತ್ರ(src->type) == 4);
        ಆದರೆ ((ggml_type_ಗಾತ್ರ(src->type) % 4) == 0) {
            ಆದರೆ (contig) {
                ಹಿಂತಿರುಗಿ ctx->device->pipeline_contig_cpy_32;
            } ಇಲ್ಲದಿದ್ದರೆ {
                ಹಿಂತಿರುಗಿ ctx->device->pipeline_cpy_32;
            }
        } ಇಲ್ಲದಿದ್ದರೆ {
            ಆದರೆ (contig) {
                ಹಿಂತಿರುಗಿ ctx->device->pipeline_contig_cpy_16;
            } ಇಲ್ಲದಿದ್ದರೆ {
                ಹಿಂತಿರುಗಿ ctx->device->pipeline_cpy_16;
            }
        }
    }

    std::cerr << "Missing CPY op for types: " << ggml_type_name(src->type) << " " << ggml_type_name(ಗಮ್ಯ) << std::ಸಾಲು_ಅಂತ್ಯ;
    GGML_ABORT("fatal error");
}

ಶೂನ್ಯ ggml_vk_cpy_ಗಮ್ಯ_contiguous(ggml_backend_vk_context * ctx, vk_context& subctx, vk_pipeline pipeline, ಸ್ಥಿರಾಂಕ ggml_tensor * tensor, ಸ್ಥಿರಾಂಕ vk_subbuffer & ಒಳ, ಸ್ಥಿರಾಂಕ vk_subbuffer & ಹೊರಹೋಗು) {
    VK_LOG_DEBUG("ggml_vk_cpy_to_contiguous((" << tensor << ", type=" << tensor->type << ", ne0=" << tensor->ne[0] << ", ne1=" << tensor->ne[1] << ", ne2=" << tensor->ne[2] << ", ne3=" << tensor->ne[3] << ", nb0=" << tensor->nb[0] << ", nb1=" << tensor->nb[1] << ", nb2=" << tensor->nb[2] << ", nb3=" << tensor->nb[3] << "), ";
    std::cerr << "buffer in size=" << ಒಳ.ಬಫರ್->size << ", buffer out size=" << ಹೊರಹೋಗು.ಬಫರ್->size << ")");

    ಸ್ಥಿರಾಂಕ uint32_t ne = ggml_nelements(tensor);
    std::array<uint32_t, 3> elements;

    ಆದರೆ (ne > 262144) {
        elements = { 512, 512, CEIL_DIV(ne, 262144) };
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (ne > 512) {
        elements = { 512, CEIL_DIV(ne, 512), 1 };
    } ಇಲ್ಲದಿದ್ದರೆ {
        elements = { ne, 1, 1 };
    }

    vk_op_unary_ತಳ್ಳು_constants pc = vk_op_unary_ತಳ್ಳು_constants_init(tensor, tensor, ne);
    pc.nb10 = 1;
    pc.nb11 = (uint32_t)tensor->ne[0];
    pc.nb12 = (uint32_t)(tensor->ne[0] * tensor->ne[1]);
    pc.nb13 = (uint32_t)(tensor->ne[0] * tensor->ne[1] * tensor->ne[2]);
    init_pushconst_fastdiv(pc);
    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { ಒಳ, ಹೊರಹೋಗು }, pc, elements);
    ggml_vk_sync_buffers(ctx, subctx);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_vk_cpy_ಗಮ್ಯ_strided(
        ggml_backend_vk_context * ctx, vk_context& subctx, vk_pipeline pipeline, ಸ್ಥಿರಾಂಕ ggml_tensor * tensor,
        ಸ್ಥಿರಾಂಕ vk_subbuffer & ಒಳ, ಸ್ಥಿರಾಂಕ vk_subbuffer & ಹೊರಹೋಗು,
        uint32_t nb10, uint32_t nb11, uint32_t nb12, uint32_t nb13) {
    VK_LOG_DEBUG("ggml_vk_cpy_to_strided((" << tensor << ", type=" << tensor->type << ", ne0=" << tensor->ne[0] << ", ne1=" << tensor->ne[1] << ", ne2=" << tensor->ne[2] << ", ne3=" << tensor->ne[3] << ", nb0=" << tensor->nb[0] << ", nb1=" << tensor->nb[1] << ", nb2=" << tensor->nb[2] << ", nb3=" << tensor->nb[3] << "), ";
    std::cerr << "dst_nb=(" << nb10 << ", " << nb11 << ", " << nb12 << ", " << nb13 << "), buffer in size=" << ಒಳ.ಬಫರ್->size << ", buffer out size=" << ಹೊರಹೋಗು.ಬಫರ್->size << ")");

    ಸ್ಥಿರಾಂಕ uint32_t ne = ggml_nelements(tensor);
    std::array<uint32_t, 3> elements;

    ಆದರೆ (ne > 262144) {
        elements = { 512, 512, CEIL_DIV(ne, 262144) };
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (ne > 512) {
        elements = { 512, CEIL_DIV(ne, 512), 1 };
    } ಇಲ್ಲದಿದ್ದರೆ {
        elements = { ne, 1, 1 };
    }

    vk_op_unary_ತಳ್ಳು_constants pc = vk_op_unary_ತಳ್ಳು_constants_init(tensor, tensor, ne);
    pc.nb10 = nb10;
    pc.nb11 = nb11;
    pc.nb12 = nb12;
    pc.nb13 = nb13;
    init_pushconst_fastdiv(pc);
    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { ಒಳ, ಹೊರಹೋಗು }, pc, elements);
    ggml_vk_sync_buffers(ctx, subctx);
}

vk_pipeline ggml_vk_get_quantize_pipeline(ggml_backend_vk_context * ctx, ggml_type type) {
    ಆಯ್ಕೆ(type) {
        ಸ್ಥಿತಿ GGML_TYPE_Q8_1:
            ಹಿಂತಿರುಗಿ ctx->device->pipeline_quantize_q8_1_x4;
        ಪೂರ್ವನಿಯೋಜಿತ:
            std::cerr << "Missing quantize pipeline for type: " << ggml_type_name(type) << std::ಸಾಲು_ಅಂತ್ಯ;
            GGML_ABORT("fatal error");
    }
}

ಶೂನ್ಯ ggml_vk_quantize_q8_1(ggml_backend_vk_context * ctx, vk_context& subctx, ಸ್ಥಿರಾಂಕ vk_subbuffer & ಒಳ, ಸ್ಥಿರಾಂಕ vk_subbuffer & ಹೊರಹೋಗು, uint32_t ne) {
    VK_LOG_DEBUG("ggml_vk_quantize_q8_1(" << "buffer in size=" << ಒಳ.ಬಫರ್->size << ", buffer out size=" << ಹೊರಹೋಗು.ಬಫರ್->size << ", " << ne << ")");

    vk_pipeline pipeline = ggml_vk_get_quantize_pipeline(ctx, GGML_TYPE_Q8_1);

    ಸ್ಥಿರಾಂಕ uint32_t ಅಂಕೆ_ಖಂಡಗಳು = CEIL_DIV(ne, pipeline->wg_denoms[0]);
    // clamp the number of elements to the max workgroup count. The shader will iterate over the total number of blocks.
    ಸ್ಥಿರಾಂಕ uint64_t max_elements = std::min<uint64_t>(uint64_t{ctx->device->properties.limits.max_Compute_Work_ಗುಂಪು_ಎಣಿಕೆ[0]} * pipeline->wg_denoms[0], std::numeric_limits<uint32_t>::max());
    ಸ್ಥಿರಾಂಕ uint32_t elements = std::min(ne, static_cast<uint32_t>(max_elements));

    ಸ್ಥಿರಾಂಕ vk_quantize_q8_1_ತಳ್ಳು_constants pc = {
        ne,
        ಅಂಕೆ_ಖಂಡಗಳು,
    };

    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { ಒಳ, ಹೊರಹೋಗು }, pc, { elements, 1, 1 });
    ggml_vk_sync_buffers(ctx, subctx);
}

ಸ್ಥಿರ vk_pipeline ggml_vk_get_64b_indexing_pipeline(ggml_backend_vk_context * ctx, vk_pipeline &pipeline) {
    GGML_ಬಳಸದ(ctx);
#ಆದರೆ defined(VK_EXT_shader_64bit_indexing)
    vk_pipeline *ptr = &pipeline;
    ಆಗ (*ptr) {
        ಆದರೆ ((*ptr)->is_64b_indexing) {
            ಹಿಂತಿರುಗಿ *ptr;
        }
        ptr = &(*ptr)->ಮುಂದಿನ;
    }
#ಆದರೆ_ಅಂತ್ಯ
    ಹಿಂತಿರುಗಿ pipeline;
}

// perf-heuristic for choosing mul_mat b type, based on shader availability
ಸ್ಥಿರ ggml_type ggml_vk_mul_mat_b_type(
        ggml_backend_vk_context * ctx, ggml_type src0_type, ಸ್ಥಿರಾಂಕ ggml_tensor * src1,
        ggml_type f16_type, ggml_prec prec, ತಾರ್ಕಿಕ mul_mat_id, ತಾರ್ಕಿಕ can_quantize_y, ತಾರ್ಕಿಕ prefer_f16_b) {
    ಸ್ವಯಂಚಾಲಿತ ಲಭ್ಯ = [&](ggml_type b) {
        ಹಿಂತಿರುಗಿ ggml_vk_get_mul_mat_mat_pipeline_map(ctx, src0_type, b, prec, mul_mat_id) != ಶೂನ್ಯಸೂಚಕ;
    };
    ಆದರೆ (can_quantize_y && ಲಭ್ಯ(GGML_TYPE_Q8_1))             ಹಿಂತಿರುಗಿ GGML_TYPE_Q8_1;
    ಆದರೆ (prefer_f16_b   && ಲಭ್ಯ(f16_type))                   ಹಿಂತಿರುಗಿ f16_type;
    ಆದರೆ (ggml_vk_dim01_contiguous(src1) && ಲಭ್ಯ(src1->type)) ಹಿಂತಿರುಗಿ src1->type;
    ಆದರೆ (src1->type == GGML_TYPE_BF16 && ಲಭ್ಯ(GGML_TYPE_F32)) ಹಿಂತಿರುಗಿ GGML_TYPE_F32;
    ಹಿಂತಿರುಗಿ f16_type;
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_vk_mul_mat_q_f16(ggml_backend_vk_context * ctx, vk_context& subctx, ಸ್ಥಿರಾಂಕ ggml_tensor * src0, ಸ್ಥಿರಾಂಕ ggml_tensor * src1, ggml_tensor * dst, ತಾರ್ಕಿಕ disable_ವಿಭಜಿಸು_k) {
    VK_LOG_DEBUG("ggml_vk_mul_mat_q_f16((" << src0 << ", name=" << src0->name << ", type=" << ggml_type_name(src0->type) << ", ne0=" << src0->ne[0] << ", ne1=" << src0->ne[1] << ", ne2=" << src0->ne[2] << ", ne3=" << src0->ne[3] << ", nb0=" << src0->nb[0] << ", nb1=" << src0->nb[1] << ", nb2=" << src0->nb[2] << ", nb3=" << src0->nb[3];
    std::cerr << "), (" << src1 << ", name=" << src1->name << ", type=" << ggml_type_name(src1->type) << ", ne0=" << src1->ne[0] << ", ne1=" << src1->ne[1] << ", ne2=" << src1->ne[2] << ", ne3=" << src1->ne[3] << ", nb0=" << src1->nb[0] << ", nb1=" << src1->nb[1] << ", nb2=" << src1->nb[2] << ", nb3=" << src1->nb[3];
    std::cerr << "), (" << dst << ", name=" << dst->name << ", type=" << ggml_type_name(dst->type) << ", ne0=" << dst->ne[0] << ", ne1=" << dst->ne[1] << ", ne2=" << dst->ne[2] << ", ne3=" << dst->ne[3] << ", nb0=" << dst->nb[0] << ", nb1=" << dst->nb[1] << ", nb2=" << dst->nb[2] << ", nb3=" << dst->nb[3];
    std::cerr << "))");
    GGML_ASSERT(ggml_vk_dim01_contiguous(src0) || src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_BF16);  // NOLINT
    GGML_ASSERT(ggml_vk_dim01_contiguous(src1) || src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16);  // NOLINT

    ಸ್ಥಿರಾಂಕ uint64_t ne00 = src0->ne[0];
    ಸ್ಥಿರಾಂಕ uint64_t ne01 = src0->ne[1];
    ಸ್ಥಿರಾಂಕ uint64_t ne02 = src0->ne[2];
    ಸ್ಥಿರಾಂಕ uint64_t ne03 = src0->ne[3];

    ಸ್ಥಿರಾಂಕ uint64_t ne10 = src1->ne[0];
    ಸ್ಥಿರಾಂಕ uint64_t ne11 = src1->ne[1];
    ಸ್ಥಿರಾಂಕ uint64_t ne12 = src1->ne[2];
    ಸ್ಥಿರಾಂಕ uint64_t ne13 = src1->ne[3];

    ಸ್ಥಿರಾಂಕ uint64_t ne21 = dst->ne[1];
    ಸ್ಥಿರಾಂಕ uint32_t stride_d = dst->nb[1] / ggml_type_ಗಾತ್ರ(dst->type);
    ಸ್ಥಿರಾಂಕ uint32_t stride_batch_d = stride_d*ne21;

    ಸ್ಥಿರಾಂಕ uint64_t r2 = ne12 / ne02;
    ಸ್ಥಿರಾಂಕ uint64_t r3 = ne13 / ne03;

    ggml_backend_vk_ಬಫರ್_context * dst_buf_ctx = (ggml_backend_vk_ಬಫರ್_context *)dst->ಬಫರ್->context;
    ggml_backend_vk_ಬಫರ್_context * src0_buf_ctx = (ggml_backend_vk_ಬಫರ್_context *)src0->ಬಫರ್->context;
    ggml_backend_vk_ಬಫರ್_context * src1_buf_ctx = (ggml_backend_vk_ಬಫರ್_context *)src1->ಬಫರ್->context;

    vk_ಬಫರ್ d_Qx = ಶೂನ್ಯಸೂಚಕ;
    size_t qx_buf_ಸ್ಥಾನಾಂತರ = 0;
    vk_ಬಫರ್ d_Qy = ಶೂನ್ಯಸೂಚಕ;
    size_t qy_buf_ಸ್ಥಾನಾಂತರ = 0;

    ತಾರ್ಕಿಕ src0_uma = ಸುಳ್ಳು;
    ತಾರ್ಕಿಕ src1_uma = ಸುಳ್ಳು;

    ಆದರೆ (ctx->device->uma) {
        ggml_vk_host_get(ctx->device, src0->ದತ್ತಾಂಶ, d_Qx, qx_buf_ಸ್ಥಾನಾಂತರ);
        ggml_vk_host_get(ctx->device, src1->ದತ್ತಾಂಶ, d_Qy, qy_buf_ಸ್ಥಾನಾಂತರ);
        src0_uma = d_Qx != ಶೂನ್ಯಸೂಚಕ;
        src1_uma = d_Qy != ಶೂನ್ಯಸೂಚಕ;
    }

    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ x_non_contig = (ctx->device->coopmat2 && src0->type == GGML_TYPE_F32) ||
                              !ggml_vk_dim01_contiguous(src0);
    // If src0 is BF16, try to use a BF16 x BF16 multiply
    ggml_type f16_type = src0->type == GGML_TYPE_BF16 ? GGML_TYPE_BF16 : GGML_TYPE_F16;

    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ is_coopmat = ctx->device->coopmat_support || ctx->device->coopmat2;
    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ can_quantize_y = (ctx->device->integer_dot_ಗುಣಲಬ್ಧ || ctx->device->coopmat_int_support) &&
                                src1->type == GGML_TYPE_F32 && ggml_is_contiguous(src1) && (ne11 * ne10) % 4 == 0;
    // coopmat shaders are fast enough that memory bandwidth reduction from f16 activations helps
    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ prefer_f16_b = is_coopmat && ggml_is_quantized(src0->type) && src1->type == GGML_TYPE_F32;

    ಸ್ಥಿರಾಂಕ ggml_type y_kernel_type = ggml_vk_mul_mat_b_type(ctx, src0->type, src1, f16_type,
                                        (ggml_prec)dst->op_params[0], ಸುಳ್ಳು, can_quantize_y, prefer_f16_b);

    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ quantize_y       = y_kernel_type == GGML_TYPE_Q8_1;
    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ y_f32_kernel     = y_kernel_type == GGML_TYPE_F32;
    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ y_needs_reformat = !quantize_y && (!ggml_vk_dim01_contiguous(src1) || y_kernel_type != src1->type);

    ಸ್ಥಿರಾಂಕ std::vector<vk_matmul_pipeline_pair>* mmp_map =
        ggml_vk_get_mul_mat_mat_pipeline_map(ctx, src0->type, y_kernel_type, (ggml_prec)dst->op_params[0]);

    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ qx_needs_dequant = mmp_map == ಶೂನ್ಯಸೂಚಕ || x_non_contig;
    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ qy_needs_dequant = y_needs_reformat;

    ಆದರೆ (qx_needs_dequant) {
        // dequant src0 to f16
        mmp_map = ggml_vk_get_mul_mat_mat_pipeline_map(ctx, f16_type, y_kernel_type, (ggml_prec)dst->op_params[0]);
    }

    GGML_ASSERT(mmp_map != ಶೂನ್ಯಸೂಚಕ);

    ಸ್ಥಿರಾಂಕ uint32_t kpad = quantize_y ? 0 : ggml_vk_align_ಗಾತ್ರ(ne10, ggml_vk_guess_matmul_pipeline_align_map(ctx, *mmp_map, ne01, ne11, ಸುಳ್ಳು));
    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ aligned = !quantize_y && ne10 == kpad && ne01 > 8 && ne11 > 8;

    vk_pipeline pipeline = ggml_vk_guess_matmul_pipeline_map(ctx, *mmp_map, ne01, ne11, aligned, ಸುಳ್ಳು);

    ಆದರೆ (ggml_nbytes(src0) > ctx->device->properties.limits.max_Storage_ಬಫರ್_Range) {
        pipeline = ggml_vk_get_64b_indexing_pipeline(ctx, pipeline);
    }

    // Reserve extra storage in the N dimension for the Y matrix, so we can avoid bounds-checking
    uint32_t padded_n = qy_needs_dequant ? ROUNDUP_POW2(ne11, pipeline->wg_denoms[1]) : ne11;
    ಸ್ಥಿರಾಂಕ uint64_t x_ne = ggml_nelements(src0);
    // 128 elements per Q8_1 x4 block
    ಸ್ಥಿರಾಂಕ uint64_t y_ne = padded_n * ne10 * ne12 * ne13;
    ಸ್ಥಿರಾಂಕ uint64_t d_ne = ggml_nelements(dst);

    ಸ್ಥಿರಾಂಕ uint32_t ವಿಭಜಿಸು_k = ggml_vk_guess_ವಿಭಜಿಸು_k(ctx, ne01, ne11, ne10, disable_ವಿಭಜಿಸು_k, pipeline);

    ಸ್ಥಿರಾಂಕ uint64_t qx_sz = ggml_type_ಗಾತ್ರ(src0->type) * x_ne / ggml_blck_ಗಾತ್ರ(src0->type);
    ಸ್ಥಿರಾಂಕ uint64_t qy_sz = ggml_type_ಗಾತ್ರ(src1->type) * y_ne / ggml_blck_ಗಾತ್ರ(src1->type);
    ಸ್ಥಿರಾಂಕ uint64_t x_sz = !qx_needs_dequant ? qx_sz : ಗಾತ್ರ(ggml_fp16_t) * x_ne;
    ಸ್ಥಿರಾಂಕ uint64_t y_sz = quantize_y ? (ggml_vk_align_ಗಾತ್ರ(y_ne, 128) * ggml_type_ಗಾತ್ರ(GGML_TYPE_Q8_1) / ggml_blck_ಗಾತ್ರ(GGML_TYPE_Q8_1)) : (y_f32_kernel ? ಗಾತ್ರ(ದಶಮಾಂಶ) * y_ne : ಗಾತ್ರ(ggml_fp16_t) * y_ne);
    ಸ್ಥಿರಾಂಕ uint64_t d_sz = ಗಾತ್ರ(ದಶಮಾಂಶ) * d_ne;

    vk_pipeline ಗಮ್ಯ_fp16_vk_0 = ಶೂನ್ಯಸೂಚಕ;
    vk_pipeline ಗಮ್ಯ_fp16_vk_1 = ಶೂನ್ಯಸೂಚಕ;
    vk_pipeline ಗಮ್ಯ_q8_1 = ಶೂನ್ಯಸೂಚಕ;

    ಆದರೆ (x_non_contig) {
        ಗಮ್ಯ_fp16_vk_0 = ggml_vk_get_cpy_pipeline(ctx, src0, ಶೂನ್ಯಸೂಚಕ, f16_type);
    } ಇಲ್ಲದಿದ್ದರೆ {
        ಗಮ್ಯ_fp16_vk_0 = ggml_vk_get_ಗಮ್ಯ_fp16(ctx, src0->type);
    }
    ಆದರೆ (y_needs_reformat) {
        ಗಮ್ಯ_fp16_vk_1 = ggml_vk_get_cpy_pipeline(ctx, src1, ಶೂನ್ಯಸೂಚಕ, y_kernel_type);
    } ಇಲ್ಲದಿದ್ದರೆ {
        ಗಮ್ಯ_fp16_vk_1 = ggml_vk_get_ಗಮ್ಯ_fp16(ctx, src1->type);
    }
    GGML_ASSERT(!qx_needs_dequant || ಗಮ್ಯ_fp16_vk_0 != ಶೂನ್ಯಸೂಚಕ);  // NOLINT
    GGML_ASSERT(!qy_needs_dequant || ಗಮ್ಯ_fp16_vk_1 != ಶೂನ್ಯಸೂಚಕ);  // NOLINT

    ಆದರೆ (quantize_y) {
        ಗಮ್ಯ_q8_1 = ggml_vk_get_quantize_pipeline(ctx, GGML_TYPE_Q8_1);
    }

    {
        ಸ್ಥಿರಾಂಕ uint64_t ವಿಭಜಿಸು_k_ಗಾತ್ರ = ವಿಭಜಿಸು_k > 1 ? d_sz * ವಿಭಜಿಸು_k : 0;
        ಆದರೆ (
                (qx_needs_dequant && x_sz > ctx->device->properties.limits.max_Storage_ಬಫರ್_Range) ||
                (qy_needs_dequant && y_sz > ctx->device->properties.limits.max_Storage_ಬಫರ್_Range) ||
                (ವಿಭಜಿಸು_k > 1 && ವಿಭಜಿಸು_k_ಗಾತ್ರ > ctx->device->properties.limits.max_Storage_ಬಫರ್_Range)) {
            GGML_ABORT("Requested preallocation size is too large");
        }
        ಆದರೆ (qx_needs_dequant && ctx->prealloc_ಗಾತ್ರ_x < x_sz) {
            ctx->prealloc_ಗಾತ್ರ_x = x_sz;
            ggml_vk_preallocate_buffers(ctx, subctx);
        }
        ಆದರೆ ((qy_needs_dequant || quantize_y) && ctx->prealloc_ಗಾತ್ರ_y < y_sz) {
            ctx->prealloc_ಗಾತ್ರ_y = y_sz;
            ggml_vk_preallocate_buffers(ctx, subctx);
        }
        ಆದರೆ (ವಿಭಜಿಸು_k > 1 && ctx->prealloc_ಗಾತ್ರ_ವಿಭಜಿಸು_k < ವಿಭಜಿಸು_k_ಗಾತ್ರ) {
            ctx->prealloc_ಗಾತ್ರ_ವಿಭಜಿಸು_k = ವಿಭಜಿಸು_k_ಗಾತ್ರ;
            ggml_vk_preallocate_buffers(ctx, subctx);
        }

        // Request descriptor sets
        ಆದರೆ (qx_needs_dequant) {
            ggml_pipeline_request_descriptor_sets(ctx, ಗಮ್ಯ_fp16_vk_0, 1);
        }
        ಆದರೆ (qy_needs_dequant) {
            ggml_pipeline_request_descriptor_sets(ctx, ಗಮ್ಯ_fp16_vk_1, 1);
        }
        ಆದರೆ (quantize_y) {
            ggml_pipeline_request_descriptor_sets(ctx, ಗಮ್ಯ_q8_1, 1);
        }
        ಆದರೆ (ವಿಭಜಿಸು_k > 1) {
            ggml_pipeline_request_descriptor_sets(ctx, ctx->device->pipeline_matmul_ವಿಭಜಿಸು_k_reduce, 1);
        }
    }

    vk_ಬಫರ್ d_D = dst_buf_ctx->dev_ಬಫರ್;
    ಸ್ಥಿರಾಂಕ uint64_t d_buf_ಸ್ಥಾನಾಂತರ = vk_tensor_ಸ್ಥಾನಾಂತರ(dst) + dst->view_offs;
    GGML_ASSERT(d_D != ಶೂನ್ಯಸೂಚಕ);
    GGML_ASSERT(d_D->size >= d_buf_ಸ್ಥಾನಾಂತರ + d_sz);
    vk_ಬಫರ್ d_X;
    uint64_t x_buf_ಸ್ಥಾನಾಂತರ = 0;
    vk_ಬಫರ್ d_Y;
    uint64_t y_buf_ಸ್ಥಾನಾಂತರ = 0;
    ಆದರೆ (!src0_uma) {
        d_Qx = src0_buf_ctx->dev_ಬಫರ್;
        qx_buf_ಸ್ಥಾನಾಂತರ = vk_tensor_ಸ್ಥಾನಾಂತರ(src0) + src0->view_offs;
        GGML_ASSERT(d_Qx != ಶೂನ್ಯಸೂಚಕ);
    }
    ಆದರೆ (!src1_uma) {
        d_Qy = src1_buf_ctx->dev_ಬಫರ್;
        qy_buf_ಸ್ಥಾನಾಂತರ = vk_tensor_ಸ್ಥಾನಾಂತರ(src1) + src1->view_offs;
        GGML_ASSERT(d_Qy != ಶೂನ್ಯಸೂಚಕ);
    }
    ಆದರೆ (qx_needs_dequant) {
        d_X = ctx->prealloc_x;
        GGML_ASSERT(d_X->size >= x_sz);
    } ಇಲ್ಲದಿದ್ದರೆ {
        d_X = d_Qx;
        x_buf_ಸ್ಥಾನಾಂತರ = qx_buf_ಸ್ಥಾನಾಂತರ;
        GGML_ASSERT(qx_sz == x_sz);
    }
    ಆದರೆ (qy_needs_dequant) {
        d_Y = ctx->prealloc_y;
        GGML_ASSERT(d_Y->size >= y_sz);
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (quantize_y) {
        d_Y = ctx->prealloc_y;
        GGML_ASSERT(d_Y->size >= CEIL_DIV(y_sz, 144) * 144);
    } ಇಲ್ಲದಿದ್ದರೆ {
        d_Y = d_Qy;
        y_buf_ಸ್ಥಾನಾಂತರ = qy_buf_ಸ್ಥಾನಾಂತರ;
        GGML_ASSERT(qy_sz == y_sz);
    }

    ಆದರೆ (x_non_contig || qx_needs_dequant) {
        ಆದರೆ (ctx->prealloc_x_need_sync) {
            ggml_vk_sync_buffers(ctx, subctx);
        }
    }

    ಆದರೆ (x_non_contig) {
        ggml_vk_cpy_ಗಮ್ಯ_contiguous(ctx, subctx, ಗಮ್ಯ_fp16_vk_0, src0, ggml_vk_subbuffer(ctx, d_Qx, qx_buf_ಸ್ಥಾನಾಂತರ), ggml_vk_subbuffer(ctx, d_X, 0));
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (qx_needs_dequant) {
        ಸ್ಥಿರಾಂಕ std::vector<uint32_t> pc = { (uint32_t)ne01, (uint32_t)ne10, (uint32_t)ne10, (uint32_t)ne10, (uint32_t)(ggml_nelements(src0)) };
        ggml_vk_dispatch_pipeline(ctx, subctx, ಗಮ್ಯ_fp16_vk_0, { vk_subbuffer{ d_Qx, qx_buf_ಸ್ಥಾನಾಂತರ, qx_sz }, vk_subbuffer{ d_X, 0, x_sz } }, pc, { (uint32_t)(x_ne), 1, 1});
        ggml_vk_sync_buffers(ctx, subctx);
    }
    ಆದರೆ (y_needs_reformat) {
        ಆದರೆ (ctx->prealloc_y_ಕೊನೆಯ_pipeline_ಬಳಸಿದ != ಗಮ್ಯ_fp16_vk_1.get() ||
            ctx->prealloc_y_ಕೊನೆಯ_tensor_ಬಳಸಿದ != src1 ||
            ctx->prealloc_y_ಕೊನೆಯ_k_padded) {
            ಆದರೆ (ctx->prealloc_y_need_sync) {
                ggml_vk_sync_buffers(ctx, subctx);
            }
            ggml_vk_cpy_ಗಮ್ಯ_contiguous(ctx, subctx, ಗಮ್ಯ_fp16_vk_1, src1, ggml_vk_subbuffer(ctx, d_Qy, qy_buf_ಸ್ಥಾನಾಂತರ), ggml_vk_subbuffer(ctx, d_Y, 0));
            ctx->prealloc_y_ಕೊನೆಯ_pipeline_ಬಳಸಿದ = ಗಮ್ಯ_fp16_vk_1.get();
            ctx->prealloc_y_ಕೊನೆಯ_tensor_ಬಳಸಿದ = src1;
            ctx->prealloc_y_ಕೊನೆಯ_k_padded = ಸುಳ್ಳು;
        }
    }
    ಆದರೆ (quantize_y) {
        ಆದರೆ (ctx->prealloc_y_ಕೊನೆಯ_pipeline_ಬಳಸಿದ != ಗಮ್ಯ_q8_1.get() ||
            ctx->prealloc_y_ಕೊನೆಯ_tensor_ಬಳಸಿದ != src1 ||
            ctx->prealloc_y_ಕೊನೆಯ_k_padded) {
            ಆದರೆ (ctx->prealloc_y_need_sync) {
                ggml_vk_sync_buffers(ctx, subctx);
            }
            ggml_vk_quantize_q8_1(ctx, subctx, ggml_vk_subbuffer(ctx, d_Qy, qy_buf_ಸ್ಥಾನಾಂತರ), ggml_vk_subbuffer(ctx, d_Y, 0), y_ne);
            ctx->prealloc_y_ಕೊನೆಯ_pipeline_ಬಳಸಿದ = ಗಮ್ಯ_q8_1.get();
            ctx->prealloc_y_ಕೊನೆಯ_tensor_ಬಳಸಿದ = src1;
            ctx->prealloc_y_ಕೊನೆಯ_k_padded = ಸುಳ್ಳು;
        }
    }

    // The partial tiles of a quantized A rely on the bound range to read zeros past the last row,
    // so the range stays exact: the strided extent of a tensor read in place, the staged size otherwise.
    ಸ್ಥಿರಾಂಕ uint64_t x_range = qx_needs_dequant ? x_sz : ggml_nbytes(src0);
    ಸ್ಥಿರಾಂಕ uint64_t y_range = (qy_needs_dequant || quantize_y) ? y_sz : ggml_nbytes(src1);

    uint32_t stride_batch_x = qx_needs_dequant ? ne00*ne01 : ggml_vk_batch_stride(src0);
    uint32_t stride_batch_y = (qy_needs_dequant || quantize_y) ? ne10*ne11 : ggml_vk_batch_stride(src1);

    ಆದರೆ (!ggml_vk_dim01_contiguous(src0) && !qx_needs_dequant) {
        stride_batch_x = src0->nb[0] / ggml_type_ಗಾತ್ರ(src0->type);
    }

    ಆದರೆ (!ggml_vk_dim01_contiguous(src1) && !qy_needs_dequant && !quantize_y) {
        stride_batch_y = src1->nb[0] / ggml_type_ಗಾತ್ರ(src1->type);
    }

    // column stride in bytes of the B buffer: raw src1 keeps its own stride, staging buffers are tightly packed
    ಸ್ಥಿರಾಂಕ uint64_t b_ಲಂಬಸಾಲು_ಬೈಟುಗಳು = (y_needs_reformat || qy_needs_dequant || quantize_y)
        ? (quantize_y ? (uint64_t)(ne10 / ggml_blck_ಗಾತ್ರ(GGML_TYPE_Q8_1)) * ggml_type_ಗಾತ್ರ(GGML_TYPE_Q8_1) : (uint64_t)ne10 * ಗಾತ್ರ(ggml_fp16_t))
        : (uint64_t)src1->nb[1];

    // compute
    ggml_vk_matmul(
        ctx, subctx, pipeline,
        { d_X, x_buf_ಸ್ಥಾನಾಂತರ, x_range }, { d_Y, y_buf_ಸ್ಥಾನಾಂತರ, y_range },
        ggml_vk_subbuffer(ctx, d_D, d_buf_ಸ್ಥಾನಾಂತರ), { ctx->prealloc_ವಿಭಜಿಸು_k, 0, d_sz * ವಿಭಜಿಸು_k },
        ne01, ne11, ne10,
        ne10, ne10, stride_d, stride_batch_x, stride_batch_y, stride_batch_d,
        ವಿಭಜಿಸು_k, ne12*ne13, ne02, ne12, r2, r3, padded_n,
        b_ಲಂಬಸಾಲು_ಬೈಟುಗಳು, (uint64_t)dst->nb[1]
    );  // NOLINT

    ಆದರೆ (x_non_contig || qx_needs_dequant) {
        ctx->prealloc_x_need_sync = ನಿಜ;
    }
    ಆದರೆ (y_needs_reformat || quantize_y) {
        ctx->prealloc_y_need_sync = ನಿಜ;
    }
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_vk_should_use_mmvq(ಸ್ಥಿರಾಂಕ vk_device& device, uint32_t m, uint32_t n, uint32_t k, ggml_type src0_type) {
    ಆದರೆ (device->mmvq_ಬಹುಲಕ == 1) {
        ಹಿಂತಿರುಗಿ ನಿಜ;
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (device->mmvq_ಬಹುಲಕ == -1) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    // q6_k only has 2-byte alignment which makes it somewhat problematic,
    // using MMVQ is only a win on Intel.
    ತಾರ್ಕಿಕ mmvq_q6 = device->vendor_id == VK_VENDOR_ID_INTEL;
    ಆದರೆ (src0_type == GGML_TYPE_Q6_K && !mmvq_q6) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    // MMVQ is generally good for batches
    ಆದರೆ (n > 1) {
        ಹಿಂತಿರುಗಿ ನಿಜ;
    }

    // Quantization overhead is not worth it for small k
    ಆಯ್ಕೆ (device->vendor_id) {
    ಸ್ಥಿತಿ VK_VENDOR_ID_NVIDIA:
        ಆದರೆ (src0_type == GGML_TYPE_Q2_0 || src0_type == GGML_TYPE_Q2_K || src0_type == GGML_TYPE_Q3_K || src0_type == GGML_TYPE_IQ1_S || src0_type == GGML_TYPE_IQ1_M) {
            ಹಿಂತಿರುಗಿ ನಿಜ;
        }

        ಆದರೆ (k <= 4096) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }

        ಆಯ್ಕೆ (src0_type) {
        ಸ್ಥಿತಿ GGML_TYPE_MXFP4:
        ಸ್ಥಿತಿ GGML_TYPE_Q8_0:
            ಹಿಂತಿರುಗಿ device->architecture == vk_device_architecture::NVIDIA_PRE_TURING;
        ಪೂರ್ವನಿಯೋಜಿತ:
            ಹಿಂತಿರುಗಿ ನಿಜ;
        }
    ಸ್ಥಿತಿ VK_VENDOR_ID_AMD:
        ಆದರೆ (k < 2048) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }

        ಆಯ್ಕೆ (src0_type) {
        ಸ್ಥಿತಿ GGML_TYPE_Q8_0:
            ಹಿಂತಿರುಗಿ device->architecture == vk_device_architecture::AMD_GCN;
        ಪೂರ್ವನಿಯೋಜಿತ:
            ಹಿಂತಿರುಗಿ ನಿಜ;
        }
    ಸ್ಥಿತಿ VK_VENDOR_ID_INTEL:
        ಆದರೆ (device->architecture == vk_device_architecture::INTEL_XE2) {
            ಆದರೆ (src0_type == GGML_TYPE_Q2_0 || src0_type == GGML_TYPE_Q2_K || src0_type == GGML_TYPE_Q3_K || src0_type == GGML_TYPE_Q6_K) {
                ಹಿಂತಿರುಗಿ ನಿಜ;
            }
        }

        ಆದರೆ (device->driver_id == vk::DriverId::eIntelProprietaryWindows) {
            // Intel Windows proprietary driver MMVQ performance for !Q2/Q3/Q6 is worse than fp16,
            // see https://github.com/ggml-org/llama.cpp/issues/17628 and
            // https://github.com/ggml-org/llama.cpp/pull/23056
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }

        ಆದರೆ (k < 2048) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }

        ಆಯ್ಕೆ (src0_type) {
        // From tests on A770 Linux, may need more tuning
        ಸ್ಥಿತಿ GGML_TYPE_Q4_0:
        ಸ್ಥಿತಿ GGML_TYPE_Q5_1:
        ಸ್ಥಿತಿ GGML_TYPE_IQ4_XS:
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        ಪೂರ್ವನಿಯೋಜಿತ:
            ಹಿಂತಿರುಗಿ ನಿಜ;
        }
    ಸ್ಥಿತಿ VK_VENDOR_ID_QUALCOMM:
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    ಪೂರ್ವನಿಯೋಜಿತ:
        ಹಿಂತಿರುಗಿ ನಿಜ;
    }

    GGML_ಬಳಸದ(m);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_vk_mul_mat_vec_q_f16(ggml_backend_vk_context * ctx, vk_context& subctx, ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_cgraph * cgraph, ಪೂರ್ಣಾಂಕ ಗಂಟು_idx, ತಾರ್ಕಿಕ ಅದಲುಬದಲು_inputs = ಸುಳ್ಳು) {
    ggml_tensor * dst = cgraph->nodes[ಗಂಟು_idx];
    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = dst->src[ಅದಲುಬದಲು_inputs ? 1 : 0];
    ಸ್ಥಿರಾಂಕ ggml_tensor * src1 = dst->src[ಅದಲುಬದಲು_inputs ? 0 : 1];

    VK_LOG_DEBUG("ggml_vk_mul_mat_vec_q_f16((" << src0 << ", name=" << src0->name << ", type=" << src0->type << ", ne0=" << src0->ne[0] << ", ne1=" << src0->ne[1] << ", ne2=" << src0->ne[2] << ", ne3=" << src0->ne[3] << ", nb0=" << src0->nb[0] << ", nb1=" << src0->nb[1] << ", nb2=" << src0->nb[2] << ", nb3=" << src0->nb[3];
    std::cerr << "), (" << src1 << ", name=" << src1->name << ", type=" << src1->type << ", ne0=" << src1->ne[0] << ", ne1=" << src1->ne[1] << ", ne2=" << src1->ne[2] << ", ne3=" << src1->ne[3] << ", nb0=" << src1->nb[0] << ", nb1=" << src1->nb[1] << ", nb2=" << src1->nb[2] << ", nb3=" << src1->nb[3];
    std::cerr << "), (" << dst << ", name=" << dst->name << ", type=" << dst->type << ", ne0=" << dst->ne[0] << ", ne1=" << dst->ne[1] << ", ne2=" << dst->ne[2] << ", ne3=" << dst->ne[3] << ", nb0=" << dst->nb[0] << ", nb1=" << dst->nb[1] << ", nb2=" << dst->nb[2] << ", nb3=" << dst->nb[3];
    std::cerr << ")),)");
    GGML_ASSERT(ggml_vk_dim01_contiguous(src0) || src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_BF16);  // NOLINT
    GGML_ASSERT(ggml_vk_dim01_contiguous(src1) || src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16);  // NOLINT

    ಸ್ಥಿರಾಂಕ uint64_t ne00 = src0->ne[0];
    ಸ್ಥಿರಾಂಕ uint64_t ne01 = src0->ne[1];
    ಸ್ಥಿರಾಂಕ uint64_t ne02 = src0->ne[2];
    ಸ್ಥಿರಾಂಕ uint64_t ne03 = src0->ne[3];

    ಸ್ಥಿರಾಂಕ uint64_t ne10 = src1->ne[0];
    ಸ್ಥಿರಾಂಕ uint64_t ne11 = src1->ne[1];
    ಸ್ಥಿರಾಂಕ uint64_t ne12 = src1->ne[2];
    ಸ್ಥಿರಾಂಕ uint64_t ne13 = src1->ne[3];

    ಸ್ಥಿರಾಂಕ uint64_t ne20 = dst->ne[ಅದಲುಬದಲು_inputs ? 1 : 0];
    ಸ್ಥಿರಾಂಕ uint64_t ne21 = dst->ne[ಅದಲುಬದಲು_inputs ? 0 : 1];
    // const uint64_t ne22 = dst->ne[2];
    // const uint64_t ne23 = dst->ne[3];

    ಸ್ಥಿರಾಂಕ uint64_t r2 = ne12 / ne02;
    ಸ್ಥಿರಾಂಕ uint64_t r3 = ne13 / ne03;

    // batch_n indicates that we need to compute a few vector results, and this assumes
    // ne12 and ne13 are 1. It overloads the batch_strides to hold the row strides.
    GGML_ASSERT(ne11 == 1 || ne12 * ne13 == 1);
    ತಾರ್ಕಿಕ batch_n = ne11 > 1;

    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ x_non_contig = !ggml_vk_dim01_contiguous(src0);

    // BF16 src1 has no mul_mat_vec B-type; widen to F32
    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ widen_bf16 = src1->type == GGML_TYPE_BF16;
    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ y_non_contig = !ggml_vk_dim01_contiguous(src1) || widen_bf16;

    ಸ್ಥಿರಾಂಕ ggml_type y_kernel_type = widen_bf16 ? GGML_TYPE_F32 : src1->type;
    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ f16_f32_kernel = y_kernel_type == GGML_TYPE_F32;
    ತಾರ್ಕಿಕ quantize_y = ctx->device->integer_dot_ಗುಣಲಬ್ಧ && src1->type == GGML_TYPE_F32 && ggml_is_contiguous(src1) && !y_non_contig && (ne11 * ne10) % 4 == 0 && ggml_vk_should_use_mmvq(ctx->device, ne01, ne11, ne10, src0->type);

    vk_pipeline ಗಮ್ಯ_fp16_vk_0 = ಶೂನ್ಯಸೂಚಕ;
    vk_pipeline ಗಮ್ಯ_fp16_vk_1 = ಶೂನ್ಯಸೂಚಕ;
    ಆದರೆ (x_non_contig) {
        ಗಮ್ಯ_fp16_vk_0 = ggml_vk_get_cpy_pipeline(ctx, src0, ಶೂನ್ಯಸೂಚಕ, src0->type);
    }
    ಆದರೆ (y_non_contig) {
        ಗಮ್ಯ_fp16_vk_1 = ggml_vk_get_cpy_pipeline(ctx, src1, ಶೂನ್ಯಸೂಚಕ, y_kernel_type);
    } ಇಲ್ಲದಿದ್ದರೆ {
        ಗಮ್ಯ_fp16_vk_1 = ggml_vk_get_ಗಮ್ಯ_fp16(ctx, src1->type);
    }

    // Check for mmq first
    vk_pipeline dmmv = quantize_y ? ggml_vk_get_dequantize_mul_mat_vec(ctx, src0->type, GGML_TYPE_Q8_1, ne11, ne20, ne00) : ಶೂನ್ಯಸೂಚಕ;
    vk_pipeline ಗಮ್ಯ_q8_1 = ಶೂನ್ಯಸೂಚಕ;

    ಆದರೆ (dmmv == ಶೂನ್ಯಸೂಚಕ) {
        // Fall back to f16 dequant mul mat
        dmmv = ggml_vk_get_dequantize_mul_mat_vec(ctx, src0->type, y_kernel_type, ne11, ne20, ne00);
        quantize_y = ಸುಳ್ಳು;
    }

    ಆದರೆ (quantize_y) {
        ಗಮ್ಯ_q8_1 = ggml_vk_get_quantize_pipeline(ctx, GGML_TYPE_Q8_1);
    }

    ಆದರೆ (ggml_nbytes(src0) > ctx->device->properties.limits.max_Storage_ಬಫರ್_Range) {
        dmmv = ggml_vk_get_64b_indexing_pipeline(ctx, dmmv);
    }

    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ qx_needs_dequant = x_non_contig;
    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ qy_needs_dequant = !quantize_y && ((src1->type != GGML_TYPE_F16 && !f16_f32_kernel) || y_non_contig);

    // Not implemented
    GGML_ASSERT(y_non_contig || !qy_needs_dequant);  // NOLINT

    GGML_ASSERT(!qx_needs_dequant || ಗಮ್ಯ_fp16_vk_0 != ಶೂನ್ಯಸೂಚಕ);  // NOLINT
    GGML_ASSERT(!qy_needs_dequant || ಗಮ್ಯ_fp16_vk_1 != ಶೂನ್ಯಸೂಚಕ);  // NOLINT
    GGML_ASSERT(dmmv != ಶೂನ್ಯಸೂಚಕ);

    ಸ್ಥಿರಾಂಕ uint64_t x_ne = ggml_nelements(src0);
    ಸ್ಥಿರಾಂಕ uint64_t y_ne = ggml_nelements(src1);

    ಸ್ಥಿರಾಂಕ uint64_t qx_sz = ggml_vk_align_ಗಾತ್ರ(ggml_type_ಗಾತ್ರ(src0->type) * x_ne / ggml_blck_ಗಾತ್ರ(src0->type), ctx->device->properties.limits.min_Storage_ಬಫರ್_ಸ್ಥಾನಾಂತರ_Alignment);
    ಸ್ಥಿರಾಂಕ uint64_t x_sz = x_non_contig ? ggml_vk_align_ಗಾತ್ರ(ggml_type_ಗಾತ್ರ(src0->type) * x_ne, ctx->device->properties.limits.min_Storage_ಬಫರ್_ಸ್ಥಾನಾಂತರ_Alignment) : qx_sz;
    ಸ್ಥಿರಾಂಕ uint64_t y_sz = quantize_y ? (ggml_vk_align_ಗಾತ್ರ(y_ne, 128) * ggml_type_ಗಾತ್ರ(GGML_TYPE_Q8_1) / ggml_blck_ಗಾತ್ರ(GGML_TYPE_Q8_1)) :
                         (f16_f32_kernel ? ಗಾತ್ರ(ದಶಮಾಂಶ) * y_ne : ಗಾತ್ರ(ggml_fp16_t) * y_ne);

    {
        ಆದರೆ (
                (qx_needs_dequant && x_sz > ctx->device->properties.limits.max_Storage_ಬಫರ್_Range) ||
                (qy_needs_dequant && y_sz > ctx->device->properties.limits.max_Storage_ಬಫರ್_Range)) {
            GGML_ABORT("Requested preallocation size is too large");
        }
        ಆದರೆ (qx_needs_dequant && ctx->prealloc_ಗಾತ್ರ_x < x_sz) {
            ctx->prealloc_ಗಾತ್ರ_x = x_sz;
            ggml_vk_preallocate_buffers(ctx, subctx);
        }
        ಆದರೆ ((qy_needs_dequant || quantize_y) && ctx->prealloc_ಗಾತ್ರ_y < y_sz) {
            ctx->prealloc_ಗಾತ್ರ_y = y_sz;
            ggml_vk_preallocate_buffers(ctx, subctx);
        }

        // Request descriptor sets
        ಆದರೆ (qx_needs_dequant) {
            ggml_pipeline_request_descriptor_sets(ctx, ಗಮ್ಯ_fp16_vk_0, 1);
        }
        ಆದರೆ (qy_needs_dequant) {
            ggml_pipeline_request_descriptor_sets(ctx, ಗಮ್ಯ_fp16_vk_1, 1);
        }
        ಆದರೆ (quantize_y) {
            ggml_pipeline_request_descriptor_sets(ctx, ಗಮ್ಯ_q8_1, 1);
        }
    }

    vk_subbuffer d_D = ggml_vk_tensor_subbuffer(ctx, cgraph->nodes[ಗಂಟು_idx + ctx->ಅಂಕೆ_additional_fused_ops]);
    vk_subbuffer d_Qx = ggml_vk_tensor_subbuffer(ctx, src0);
    vk_subbuffer d_Qy = ggml_vk_tensor_subbuffer(ctx, src1);
    vk_subbuffer d_X, d_Y;

    ಆದರೆ (qx_needs_dequant) {
        d_X = { ctx->prealloc_x, 0, ctx->prealloc_x->size };
    } ಇಲ್ಲದಿದ್ದರೆ {
        d_X = d_Qx;
        GGML_ASSERT(qx_sz == x_sz);
    }
    ಆದರೆ (qy_needs_dequant || quantize_y) {
        d_Y = { ctx->prealloc_y, 0, ctx->prealloc_y->size };
    } ಇಲ್ಲದಿದ್ದರೆ {
        d_Y = d_Qy;
    }

    ಆದರೆ (x_non_contig) {
        ಆದರೆ (ctx->prealloc_x_need_sync) {
            ggml_vk_sync_buffers(ctx, subctx);
        }

        GGML_ASSERT(x_sz == ggml_vk_align_ಗಾತ್ರ(ggml_type_ಗಾತ್ರ(src0->type) * x_ne, ctx->device->properties.limits.min_Storage_ಬಫರ್_ಸ್ಥಾನಾಂತರ_Alignment));
        ggml_vk_cpy_ಗಮ್ಯ_contiguous(ctx, subctx, ಗಮ್ಯ_fp16_vk_0, src0, d_Qx, d_X);
    }
    ಆದರೆ (y_non_contig) {
        GGML_ASSERT(y_sz == ggml_type_ಗಾತ್ರ(y_kernel_type) * y_ne);
        ಆದರೆ (ctx->prealloc_y_ಕೊನೆಯ_pipeline_ಬಳಸಿದ != ಗಮ್ಯ_fp16_vk_1.get() ||
            ctx->prealloc_y_ಕೊನೆಯ_tensor_ಬಳಸಿದ != src1 ||
            ctx->prealloc_y_ಕೊನೆಯ_k_padded) {
            ಆದರೆ (ctx->prealloc_y_need_sync) {
                ggml_vk_sync_buffers(ctx, subctx);
            }
            ggml_vk_cpy_ಗಮ್ಯ_contiguous(ctx, subctx, ಗಮ್ಯ_fp16_vk_1, src1, d_Qy, d_Y);
            ctx->prealloc_y_ಕೊನೆಯ_pipeline_ಬಳಸಿದ = ಗಮ್ಯ_fp16_vk_1.get();
            ctx->prealloc_y_ಕೊನೆಯ_tensor_ಬಳಸಿದ = src1;
            ctx->prealloc_y_ಕೊನೆಯ_k_padded = ಸುಳ್ಳು;
        }
    }
    ಆದರೆ (quantize_y) {
        ಆದರೆ (ctx->prealloc_y_ಕೊನೆಯ_pipeline_ಬಳಸಿದ != ಗಮ್ಯ_q8_1.get() ||
            ctx->prealloc_y_ಕೊನೆಯ_tensor_ಬಳಸಿದ != src1 ||
            ctx->prealloc_y_ಕೊನೆಯ_k_padded) {
            ಆದರೆ (ctx->prealloc_y_need_sync) {
                ggml_vk_sync_buffers(ctx, subctx);
            }
            ggml_vk_quantize_q8_1(ctx, subctx, d_Qy, d_Y, y_ne);
            ctx->prealloc_y_ಕೊನೆಯ_pipeline_ಬಳಸಿದ = ಗಮ್ಯ_q8_1.get();
            ctx->prealloc_y_ಕೊನೆಯ_tensor_ಬಳಸಿದ = src1;
            ctx->prealloc_y_ಕೊನೆಯ_k_padded = ಸುಳ್ಳು;
        }
    }

    // For batch_n, the A matrix is the same for each batch, and B/D use the row stride as the batch stride
    uint32_t stride_batch_x = batch_n ? 0 : (qx_needs_dequant ? ne00*ne01 : ggml_vk_batch_stride(src0));
    uint32_t stride_batch_y = batch_n ? ne10 : ((qy_needs_dequant || quantize_y) ? ne10*ne11 : ggml_vk_batch_stride(src1));
    uint32_t stride_batch_d = batch_n ? ne20 : (ne20*ne21);

    ಆದರೆ (!ggml_vk_dim01_contiguous(src0) && !qx_needs_dequant) {
        stride_batch_x = src0->nb[0] / ggml_type_ಗಾತ್ರ(src0->type);
    }

    ಆದರೆ (!ggml_vk_dim01_contiguous(src1) && !qy_needs_dequant) {
        stride_batch_y = src1->nb[0] / ggml_type_ಗಾತ್ರ(src1->type);
    }

    ಸ್ಥಿರಾಂಕ uint32_t max_groups_x = ctx->device->properties.limits.max_Compute_Work_ಗುಂಪು_ಎಣಿಕೆ[0];

    uint32_t groups_x = ne01;
    uint32_t groups_z = 1;

    ಆದರೆ (ne01 > max_groups_x) {
        groups_z = 64;
        groups_x = CEIL_DIV(groups_x, groups_z);
    }

    uint32_t fusion_flags = 0;

    vk_subbuffer d_F0 = d_D;
    ಆದರೆ (ctx->ಅಂಕೆ_additional_fused_ops > 0) {
        ಸ್ಥಿರಾಂಕ ggml_tensor * ಕೂಡಿಸು_2 = cgraph->nodes[ಗಂಟು_idx + 1];
        ಸ್ಥಿರಾಂಕ ggml_tensor * bias = ಕೂಡಿಸು_2->src[0] == dst ? ಕೂಡಿಸು_2->src[1] : ಕೂಡಿಸು_2->src[0];

        d_F0 = ggml_vk_tensor_subbuffer(ctx, bias);
        fusion_flags |= MAT_VEC_FUSION_FLAGS_BIAS0;
    }

    vk_subbuffer d_F1 = d_D;
    ಆದರೆ (ctx->ಅಂಕೆ_additional_fused_ops == 2) {
        ಸ್ಥಿರಾಂಕ ggml_tensor * ಕೂಡಿಸು_2 = cgraph->nodes[ಗಂಟು_idx + 2];
        ಸ್ಥಿರಾಂಕ ggml_tensor * bias = ಕೂಡಿಸು_2->src[0] == cgraph->nodes[ಗಂಟು_idx + 1] ? ಕೂಡಿಸು_2->src[1] : ಕೂಡಿಸು_2->src[0];

        d_F1 = ggml_vk_tensor_subbuffer(ctx, bias);
        fusion_flags |= MAT_VEC_FUSION_FLAGS_BIAS1;
    }

    ggml_pipeline_request_descriptor_sets(ctx, dmmv, CEIL_DIV(ne12 * ne13, ctx->device->properties.limits.max_Compute_Work_ಗುಂಪು_ಎಣಿಕೆ[1]));

    uint32_t ತಳ_work_ಗುಂಪು_y = 0;
    ಆಗ (ತಳ_work_ಗುಂಪು_y < ne12 * ne13) {

        uint32_t groups_y = std::min((uint32_t)(ne12 * ne13) - ತಳ_work_ಗುಂಪು_y, ctx->device->properties.limits.max_Compute_Work_ಗುಂಪು_ಎಣಿಕೆ[1]);
        ಸ್ಥಿರಾಂಕ vk_mat_vec_ತಳ್ಳು_constants pc = {
            (uint32_t)ne00, (uint32_t)ne10, (uint32_t)ne10, (uint32_t)ne01,
            stride_batch_x, stride_batch_y, stride_batch_d,
            fusion_flags, ತಳ_work_ಗುಂಪು_y,
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
        ತಳ_work_ಗುಂಪು_y += groups_y;
    }

    ಆದರೆ (x_non_contig) {
        ctx->prealloc_x_need_sync = ನಿಜ;
    }
    ಆದರೆ (y_non_contig || quantize_y) {
        ctx->prealloc_y_need_sync = ನಿಜ;
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_vk_mul_mat_vec_p021_f16_f32(ggml_backend_vk_context * ctx, vk_context& subctx, ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_cgraph * cgraph, ಪೂರ್ಣಾಂಕ ಗಂಟು_idx) {
    ggml_tensor * dst = cgraph->nodes[ಗಂಟು_idx];
    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = dst->src[0];
    ಸ್ಥಿರಾಂಕ ggml_tensor * src1 = dst->src[1];
    VK_LOG_DEBUG("ggml_vk_mul_mat_p021_f16_f32(" << src0 << ", name=" << src0->name << ", type=" << src0->type << ", ne0=" << src0->ne[0] << ", ne1=" << src0->ne[1] << ", ne2=" << src0->ne[2] << ", ne3=" << src0->ne[3] << ", nb0=" << src0->nb[0] << ", nb1=" << src0->nb[1] << ", nb2=" << src0->nb[2] << ", nb3=" << src0->nb[3];
    std::cerr << "), (" << src1 << ", name=" << src1->name << ", type=" << src1->type << ", ne0=" << src1->ne[0] << ", ne1=" << src1->ne[1] << ", ne2=" << src1->ne[2] << ", ne3=" << src1->ne[3] << ", nb0=" << src1->nb[0] << ", nb1=" << src1->nb[1] << ", nb2=" << src1->nb[2] << ", nb3=" << src1->nb[3];
    std::cerr << "), (" << dst << ", name=" << dst->name << ", type=" << dst->type << ", ne0=" << dst->ne[0] << ", ne1=" << dst->ne[1] << ", ne2=" << dst->ne[2] << ", ne3=" << dst->ne[3] << ", nb0=" << dst->nb[0] << ", nb1=" << dst->nb[1] << ", nb2=" << dst->nb[2] << ", nb3=" << dst->nb[3];
    std::cerr << "))");
    GGML_ASSERT(ggml_is_permuted(src0) && ggml_is_permuted(src1));
    GGML_ASSERT(src0->nb[0] <= src0->nb[1] && src0->nb[2] <= src0->nb[3]);  // NOLINT
    GGML_ASSERT(src1->nb[0] <= src1->nb[1] && src1->nb[2] <= src1->nb[3]);  // NOLINT
    GGML_ASSERT(src0->type == GGML_TYPE_F16);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);

    ಸ್ಥಿರಾಂಕ uint64_t ne00 = src0->ne[0];
    ಸ್ಥಿರಾಂಕ uint64_t ne01 = src0->ne[1];
    ಸ್ಥಿರಾಂಕ uint64_t ne02 = src0->ne[2];
    // const uint64_t ne03 = src0->ne[3];

    //const uint64_t ne10 = src1->ne[0];
    ಸ್ಥಿರಾಂಕ uint64_t ne11 = src1->ne[1];
    ಸ್ಥಿರಾಂಕ uint64_t ne12 = src1->ne[2];
    // const uint64_t ne13 = src1->ne[3];

    GGML_ASSERT(ne11 == 1);

    // With grouped query attention there are > 1 Q matrices per K, V matrix.
    uint32_t gqa_ಅನುಪಾತ = (uint32_t)ne12 / (uint32_t)ne02;
    ಆದರೆ (gqa_ಅನುಪಾತ > 8 || gqa_ಅನುಪಾತ == 0 || ne12 != ne02 * gqa_ಅನುಪಾತ) {
        gqa_ಅನುಪಾತ = 1;
    }

    vk_pipeline pipeline = ctx->device->pipeline_mul_mat_vec_p021_f16_f32[gqa_ಅನುಪಾತ - 1];

    ಆದರೆ (ggml_nbytes(src0) > ctx->device->properties.limits.max_Storage_ಬಫರ್_Range) {
        pipeline = ggml_vk_get_64b_indexing_pipeline(ctx, pipeline);
    }

    {
        // Request descriptor sets
        ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);
    }

    vk_subbuffer d_D = ggml_vk_tensor_subbuffer(ctx, cgraph->nodes[ಗಂಟು_idx + ctx->ಅಂಕೆ_additional_fused_ops], ನಿಜ);
    vk_subbuffer d_Qx = ggml_vk_tensor_subbuffer(ctx, src0);
    vk_subbuffer d_Qy = ggml_vk_tensor_subbuffer(ctx, src1, ನಿಜ);

    vk_subbuffer d_F0 = d_D;

    uint32_t fusion_flags = 0;

    ಆದರೆ (ctx->ಅಂಕೆ_additional_fused_ops > 0) {
        ಸ್ಥಿರಾಂಕ ggml_tensor * ಕೂಡಿಸು_2 = cgraph->nodes[ಗಂಟು_idx + 1];
        ಸ್ಥಿರಾಂಕ ggml_tensor * bias = ಕೂಡಿಸು_2->src[0] == dst ? ಕೂಡಿಸು_2->src[1] : ಕೂಡಿಸು_2->src[0];

        d_F0 = ggml_vk_tensor_subbuffer(ctx, bias);
        fusion_flags |= MAT_VEC_FUSION_FLAGS_BIAS0;
    }

    vk_subbuffer d_F1 = d_D;
    ಆದರೆ (ctx->ಅಂಕೆ_additional_fused_ops > 1) {
        ಸ್ಥಿರಾಂಕ ggml_tensor * bias = cgraph->nodes[ಗಂಟು_idx + 2]->src[1];

        d_F1 = ggml_vk_tensor_subbuffer(ctx, bias);
        fusion_flags |= MAT_VEC_FUSION_FLAGS_BIAS1;
    }

    // compute

    vk_mat_vec_p021_ತಳ್ಳು_constants pc = {
        (uint32_t)ne00, (uint32_t)ne01, (uint32_t)ne02, (uint32_t)ne12,
        0, 0, fusion_flags
    };

    init_pushconst_tensor_offsets(ctx, pc, src0, src1, ಶೂನ್ಯಸೂಚಕ, ಶೂನ್ಯಸೂಚಕ, cgraph->nodes[ಗಂಟು_idx + ctx->ಅಂಕೆ_additional_fused_ops]);

    uint32_t workgroups_z = (uint32_t)ne12;
    // When gqa_ratio > 1, each invocation does multiple rows and we can launch fewer workgroups
    ಆದರೆ (gqa_ಅನುಪಾತ > 1) {
        workgroups_z /= gqa_ಅನುಪಾತ;
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

ಸ್ಥಿರ ಶೂನ್ಯ ggml_vk_mul_mat_vec_nc_f16_f32(ggml_backend_vk_context * ctx, vk_context& subctx, ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_cgraph * cgraph, ಪೂರ್ಣಾಂಕ ಗಂಟು_idx) {
    ggml_tensor * dst = cgraph->nodes[ಗಂಟು_idx];
    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = dst->src[0];
    ಸ್ಥಿರಾಂಕ ggml_tensor * src1 = dst->src[1];
    VK_LOG_DEBUG("ggml_vk_mul_mat_nc_f16_f32((" << src0 << ", name=" << src0->name << ", type=" << src0->type << ", ne0=" << src0->ne[0] << ", ne1=" << src0->ne[1] << ", ne2=" << src0->ne[2] << ", ne3=" << src0->ne[3] << ", nb0=" << src0->nb[0] << ", nb1=" << src0->nb[1] << ", nb2=" << src0->nb[2] << ", nb3=" << src0->nb[3];
    std::cerr << "), (" << src1 << ", name=" << src1->name << ", type=" << src1->type << ", ne0=" << src1->ne[0] << ", ne1=" << src1->ne[1] << ", ne2=" << src1->ne[2] << ", ne3=" << src1->ne[3] << ", nb0=" << src1->nb[0] << ", nb1=" << src1->nb[1] << ", nb2=" << src1->nb[2] << ", nb3=" << src1->nb[3];
    std::cerr << "), (" << dst << ", name=" << dst->name << ", type=" << dst->type << ", ne0=" << dst->ne[0] << ", ne1=" << dst->ne[1] << ", ne2=" << dst->ne[2] << ", ne3=" << dst->ne[3] << ", nb0=" << dst->nb[0] << ", nb1=" << dst->nb[1] << ", nb2=" << dst->nb[2] << ", nb3=" << dst->nb[3];
    std::cerr << "))");
    GGML_ASSERT(!ggml_is_transposed(src0));
    GGML_ASSERT(!ggml_is_transposed(src1));
    GGML_ASSERT(!ggml_is_permuted(src0));
    GGML_ASSERT(src0->type == GGML_TYPE_F16);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);

    ಸ್ಥಿರಾಂಕ uint64_t ne00 = src0->ne[0];
    ಸ್ಥಿರಾಂಕ uint64_t ne01 = src0->ne[1];
    ಸ್ಥಿರಾಂಕ uint64_t ne02 = src0->ne[2];
    ಸ್ಥಿರಾಂಕ uint64_t ne03 = src0->ne[3];

    ಸ್ಥಿರಾಂಕ uint64_t nb01 = src0->nb[1];
    ಸ್ಥಿರಾಂಕ uint64_t nb02 = src0->nb[2];

    ಸ್ಥಿರಾಂಕ uint64_t nb12 = src1->nb[2];

    // const uint64_t ne10 = src1->ne[0];
    ಸ್ಥಿರಾಂಕ uint64_t ne11 = src1->ne[1];
    ಸ್ಥಿರಾಂಕ uint64_t ne12 = src1->ne[2];
    // const uint64_t ne13 = src1->ne[3];

    ಸ್ಥಿರಾಂಕ uint32_t nb03 = (uint32_t)(src0->nb[3] / ಗಾತ್ರ(ggml_fp16_t));
    ಸ್ಥಿರಾಂಕ uint32_t nb13 = (uint32_t)(src1->nb[3] / ಗಾತ್ರ(ದಶಮಾಂಶ));
    ಸ್ಥಿರಾಂಕ uint32_t nb23 = (uint32_t)(dst->nb[3] / ಗಾತ್ರ(ದಶಮಾಂಶ));

    GGML_ASSERT(ne11 == 1);
    GGML_ASSERT(src0->ne[3] == src1->ne[3]); // checked in supports_op

    ಸ್ಥಿರಾಂಕ uint32_t ಅಡ್ಡಸಾಲು_stride_x = nb01 / ಗಾತ್ರ(ggml_fp16_t);
    ಸ್ಥಿರಾಂಕ uint32_t channel_stride_x = nb02 / ಗಾತ್ರ(ggml_fp16_t);
    ಸ್ಥಿರಾಂಕ uint32_t channel_stride_y = nb12 / ಗಾತ್ರ(ದಶಮಾಂಶ);

    vk_pipeline pipeline = ctx->device->pipeline_mul_mat_vec_nc_f16_f32;
    ಆದರೆ (ggml_nbytes(src0) > ctx->device->properties.limits.max_Storage_ಬಫರ್_Range) {
        pipeline = ggml_vk_get_64b_indexing_pipeline(ctx, pipeline);
    }

    {
        // Request descriptor sets
        ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);
    }

    vk_subbuffer d_D = ggml_vk_tensor_subbuffer(ctx, cgraph->nodes[ಗಂಟು_idx + ctx->ಅಂಕೆ_additional_fused_ops], ನಿಜ);
    vk_subbuffer d_Qx = ggml_vk_tensor_subbuffer(ctx, src0);
    vk_subbuffer d_Qy = ggml_vk_tensor_subbuffer(ctx, src1, ನಿಜ);
    vk_subbuffer d_F0 = d_D;

    uint32_t fusion_flags = 0;

    ಆದರೆ (ctx->ಅಂಕೆ_additional_fused_ops > 0) {
        ಸ್ಥಿರಾಂಕ ggml_tensor * ಕೂಡಿಸು_2 = cgraph->nodes[ಗಂಟು_idx + 1];
        ಸ್ಥಿರಾಂಕ ggml_tensor * bias = ಕೂಡಿಸು_2->src[0] == dst ? ಕೂಡಿಸು_2->src[1] : ಕೂಡಿಸು_2->src[0];

        d_F0 = ggml_vk_tensor_subbuffer(ctx, bias);
        fusion_flags |= MAT_VEC_FUSION_FLAGS_BIAS0;
    }

    vk_subbuffer d_F1 = d_D;
    ಆದರೆ (ctx->ಅಂಕೆ_additional_fused_ops > 1) {
        ಸ್ಥಿರಾಂಕ ggml_tensor * bias = cgraph->nodes[ಗಂಟು_idx + 2]->src[1];

        d_F1 = ggml_vk_tensor_subbuffer(ctx, bias);
        fusion_flags |= MAT_VEC_FUSION_FLAGS_BIAS1;
    }

    // compute
    vk_mat_vec_nc_ತಳ್ಳು_constants pc = {
        (uint32_t)ne00, (uint32_t)ne01,
        ಅಡ್ಡಸಾಲು_stride_x, channel_stride_x, channel_stride_y,
        (uint32_t)(ne12 / ne02), (uint32_t)ne12,
        0, 0,
        nb03, nb13, nb23, fusion_flags
    };

    init_pushconst_tensor_offsets(ctx, pc, src0, src1, ಶೂನ್ಯಸೂಚಕ, ಶೂನ್ಯಸೂಚಕ, cgraph->nodes[ಗಂಟು_idx + ctx->ಅಂಕೆ_additional_fused_ops]);

    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline,
        {
            d_Qx,
            d_Qy,
            d_D,
            d_F0,
            d_F1,
        }, pc, { (uint32_t)ne03, (uint32_t)ne01, (uint32_t)ne12 });
}

ಸ್ಥಿರ ಪೂರ್ಣಾಂಕ ggml_vk_fwht_pipeline_idx(int64_t n) {
    ಆಯ್ಕೆ (n) {
        ಸ್ಥಿತಿ 64:  ಹಿಂತಿರುಗಿ 0;
        ಸ್ಥಿತಿ 128: ಹಿಂತಿರುಗಿ 1;
        ಸ್ಥಿತಿ 256: ಹಿಂತಿರುಗಿ 2;
        ಸ್ಥಿತಿ 512: ಹಿಂತಿರುಗಿ 3;
        ಪೂರ್ವನಿಯೋಜಿತ:  ಹಿಂತಿರುಗಿ -1;
    }
}

ತಾರ್ಕಿಕ ggml_vk_can_use_fwht(ಸ್ಥಿರಾಂಕ ggml_backend_vk_context * ctx, ಸ್ಥಿರಾಂಕ ggml_tensor * src1, ಸ್ಥಿರಾಂಕ ggml_tensor * dst) {
    ಆದರೆ (ctx->ಅಂಕೆ_additional_fused_ops != 0) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಆದರೆ (ggml_get_op_params_i32(dst, 1) != GGML_HINT_SRC0_IS_HADAMARD) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ idx = ggml_vk_fwht_pipeline_idx(src1->ne[0]);
    ಆದರೆ (idx < 0 || ctx->device->pipeline_fwht_f32[idx] == ಶೂನ್ಯಸೂಚಕ) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಆದರೆ (src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಆದರೆ (!ggml_is_contiguous(src1)) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
    GGML_ASSERT(ggml_is_contiguous(dst));

    ಹಿಂತಿರುಗಿ ನಿಜ;
}

ಶೂನ್ಯ ggml_vk_fwht(ggml_backend_vk_context * ctx, vk_context& subctx, ಸ್ಥಿರಾಂಕ ggml_tensor * src, ggml_tensor * dst) {
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ idx = ggml_vk_fwht_pipeline_idx(src->ne[0]);
    vk_pipeline pipeline = ctx->device->pipeline_fwht_f32[idx];

    ಸ್ಥಿರಾಂಕ uint32_t ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_workgroup = 4;
    ಸ್ಥಿರಾಂಕ uint32_t n_ಅಡ್ಡಸಾಲುಗಳು = (uint32_t)ggml_nrows(src);
    ಸ್ಥಿರಾಂಕ uint32_t max_workgroups_x = ctx->device->properties.limits.max_Compute_Work_ಗುಂಪು_ಎಣಿಕೆ[0];

    ಸ್ಥಿರಾಂಕ uint32_t ಒಟ್ಟು_workgroups = CEIL_DIV(n_ಅಡ್ಡಸಾಲುಗಳು, ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_workgroup);
    ಸ್ಥಿರಾಂಕ uint32_t workgroups_x = std::min(ಒಟ್ಟು_workgroups, max_workgroups_x);
    ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);

    ಸ್ಥಿರಾಂಕ vk_subbuffer src_buf = ggml_vk_tensor_subbuffer(ctx, src, ನಿಜ);
    ಸ್ಥಿರಾಂಕ vk_subbuffer dst_buf = ggml_vk_tensor_subbuffer(ctx, dst, ನಿಜ);

    vk_op_fwht_ತಳ್ಳು_constants pc = {
        n_ಅಡ್ಡಸಾಲುಗಳು,
        0,
        0,
        1.0f / std::ವರ್ಗಮೂಲ((ದಶಮಾಂಶ)src->ne[0]),
    };
    init_pushconst_tensor_offsets(ctx, pc, src, ಶೂನ್ಯಸೂಚಕ, ಶೂನ್ಯಸೂಚಕ, ಶೂನ್ಯಸೂಚಕ, dst);

    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { src_buf, dst_buf }, pc, { workgroups_x, 1, 1 });
}

ಸ್ಥಿರ uint32_t ggml_vk_nb_elem(ಸ್ಥಿರಾಂಕ ggml_tensor * t, ಪೂರ್ಣಾಂಕ i) {
    ಹಿಂತಿರುಗಿ (uint32_t)(t->nb[i] / ggml_type_ಗಾತ್ರ(t->type));
}

ಶೂನ್ಯ ggml_vk_dsv4_hc_comb(ggml_backend_vk_context * ctx, vk_context& subctx, ಸ್ಥಿರಾಂಕ ggml_tensor * mixes, ಸ್ಥಿರಾಂಕ ggml_tensor * scale, ಸ್ಥಿರಾಂಕ ggml_tensor * ತಳ, ggml_tensor * dst) {
    VK_LOG_DEBUG("ggml_vk_dsv4_hc_comb(" << mixes << ", " << scale << ", " << ತಳ << ", " << dst << ")");

    vk_pipeline pipeline = ctx->device->pipeline_dsv4_hc_comb_f32;
    GGML_ASSERT(pipeline != ಶೂನ್ಯಸೂಚಕ);

    ಸ್ಥಿರಾಂಕ uint32_t n_ಟೋಕನುಗಳು = (uint32_t)mixes->ne[1];

    ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);

    ಸ್ಥಿರಾಂಕ vk_subbuffer mixes_buf = ggml_vk_tensor_subbuffer(ctx, mixes, ನಿಜ);
    ಸ್ಥಿರಾಂಕ vk_subbuffer scale_buf = ggml_vk_tensor_subbuffer(ctx, scale, ನಿಜ);
    ಸ್ಥಿರಾಂಕ vk_subbuffer ತಳ_buf  = ggml_vk_tensor_subbuffer(ctx, ತಳ,  ನಿಜ);
    ಸ್ಥಿರಾಂಕ vk_subbuffer dst_buf   = ggml_vk_tensor_subbuffer(ctx, dst,   ನಿಜ);

    vk_op_dsv4_hc_comb_ತಳ್ಳು_constants pc = {
        n_ಟೋಕನುಗಳು,
        ggml_vk_nb_elem(mixes, 0), ggml_vk_nb_elem(mixes, 1),
        ggml_vk_nb_elem(scale, 0),
        ggml_vk_nb_elem(ತಳ,  0),
        ggml_vk_nb_elem(dst,   0), ggml_vk_nb_elem(dst, 1), ggml_vk_nb_elem(dst, 2),
        0, 0, 0, 0,
        ggml_get_op_params_f32(dst, 0),
        (uint32_t)ggml_get_op_params_i32(dst, 1),
    };
    init_pushconst_tensor_offsets(ctx, pc, mixes, scale, ತಳ, ಶೂನ್ಯಸೂಚಕ, dst);

    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { mixes_buf, scale_buf, ತಳ_buf, dst_buf }, pc, { n_ಟೋಕನುಗಳು, 1, 1 });
}

ಶೂನ್ಯ ggml_vk_dsv4_hc_pre(ggml_backend_vk_context * ctx, vk_context& subctx, ಸ್ಥಿರಾಂಕ ggml_tensor * x, ಸ್ಥಿರಾಂಕ ggml_tensor * weights, ggml_tensor * dst) {
    VK_LOG_DEBUG("ggml_vk_dsv4_hc_pre(" << x << ", " << weights << ", " << dst << ")");

    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ scale = ggml_get_op_params_f32(dst, 0);
    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ  gated = ggml_get_op_params_i32(dst, 1) != 0;

    vk_pipeline pipeline = gated ? ctx->device->pipeline_dsv4_hc_pre_gated_f32 : ctx->device->pipeline_dsv4_hc_pre_f32;
    GGML_ASSERT(pipeline != ಶೂನ್ಯಸೂಚಕ);

    ಸ್ಥಿರಾಂಕ uint32_t n_embd   = (uint32_t)x->ne[0];
    ಸ್ಥಿರಾಂಕ uint32_t n_ಟೋಕನುಗಳು = (uint32_t)x->ne[2];

    ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);

    ಸ್ಥಿರಾಂಕ vk_subbuffer x_buf = ggml_vk_tensor_subbuffer(ctx, x,       ನಿಜ);
    ಸ್ಥಿರಾಂಕ vk_subbuffer w_buf = ggml_vk_tensor_subbuffer(ctx, weights, ನಿಜ);
    ಸ್ಥಿರಾಂಕ vk_subbuffer d_buf = ggml_vk_tensor_subbuffer(ctx, dst,     ನಿಜ);

    vk_op_dsv4_hc_pre_ತಳ್ಳು_constants pc = {
        n_embd, n_ಟೋಕನುಗಳು,
        ggml_vk_nb_elem(x, 0), ggml_vk_nb_elem(x, 1), ggml_vk_nb_elem(x, 2),
        ggml_vk_nb_elem(weights, 0), ggml_vk_nb_elem(weights, 1), ggml_vk_nb_elem(weights, 2),
        ggml_vk_nb_elem(dst, 0), ggml_vk_nb_elem(dst, 1),
        0, 0, 0,
        scale,
    };
    init_pushconst_tensor_offsets(ctx, pc, x, weights, ಶೂನ್ಯಸೂಚಕ, ಶೂನ್ಯಸೂಚಕ, dst);

    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { x_buf, w_buf, d_buf }, pc, { n_embd, n_ಟೋಕನುಗಳು, 1 });
}

ಶೂನ್ಯ ggml_vk_dsv4_hc_post(ggml_backend_vk_context * ctx, vk_context& subctx, ಸ್ಥಿರಾಂಕ ggml_tensor * x, ಸ್ಥಿರಾಂಕ ggml_tensor * residual, ಸ್ಥಿರಾಂಕ ggml_tensor * post, ಸ್ಥಿರಾಂಕ ggml_tensor * comb, ggml_tensor * dst, ಸ್ಥಿರಾಂಕ ggml_tensor * gate_scale_ಒಳ) {
    VK_LOG_DEBUG("ggml_vk_dsv4_hc_post(" << x << ", " << residual << ", " << post << ", " << comb << ", " << dst << ")");

    vk_pipeline pipeline = comb ? ctx->device->pipeline_dsv4_hc_post_f32 : ctx->device->pipeline_dsv4_hc_post_nocomb_f32;
    GGML_ASSERT(pipeline != ಶೂನ್ಯಸೂಚಕ);

    ಸ್ಥಿರಾಂಕ uint32_t n_embd   = (uint32_t)x->ne[0];
    ಸ್ಥಿರಾಂಕ uint32_t n_ಟೋಕನುಗಳು = (uint32_t)x->ne[1];

    ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);

    ಸ್ಥಿರಾಂಕ vk_subbuffer x_buf = ggml_vk_tensor_subbuffer(ctx, x,        ನಿಜ);
    ಸ್ಥಿರಾಂಕ vk_subbuffer r_buf = ggml_vk_tensor_subbuffer(ctx, residual, ನಿಜ);
    // with a fused gate, post is scale(sigmoid(scale(p_src))) and the shader applies it to p_src
    ಸ್ಥಿರಾಂಕ ggml_tensor * p_src = gate_scale_ಒಳ ? gate_scale_ಒಳ->src[0] : post;
    ಸ್ಥಿರಾಂಕ vk_subbuffer p_buf = ggml_vk_tensor_subbuffer(ctx, p_src,    ನಿಜ);
    ಸ್ಥಿರಾಂಕ vk_subbuffer c_buf = comb ? ggml_vk_tensor_subbuffer(ctx, comb, ನಿಜ) : x_buf;
    ಸ್ಥಿರಾಂಕ vk_subbuffer d_buf = ggml_vk_tensor_subbuffer(ctx, dst,      ನಿಜ);

    vk_op_dsv4_hc_post_ತಳ್ಳು_constants pc = {
        n_embd, n_ಟೋಕನುಗಳು,
        ggml_vk_nb_elem(x, 0), ggml_vk_nb_elem(x, 1),
        ggml_vk_nb_elem(residual, 0), ggml_vk_nb_elem(residual, 1), ggml_vk_nb_elem(residual, 2),
        ggml_vk_nb_elem(p_src, 0), ggml_vk_nb_elem(p_src, 1),
        comb ? ggml_vk_nb_elem(comb, 0) : 0, comb ? ggml_vk_nb_elem(comb, 1) : 0, comb ? ggml_vk_nb_elem(comb, 2) : 0,
        ggml_vk_nb_elem(dst,  0), ggml_vk_nb_elem(dst,  1), ggml_vk_nb_elem(dst,  2),
        0, 0, 0, 0, 0,
        gate_scale_ಒಳ ? 1u : 0u,
        gate_scale_ಒಳ ? ggml_get_op_params_f32(gate_scale_ಒಳ, 0) : 1.0f,
        gate_scale_ಒಳ ? ggml_get_op_params_f32(post, 0) : 1.0f,
    };
    init_pushconst_tensor_offsets(ctx, pc, x, residual, p_src, comb, dst);

    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { x_buf, r_buf, p_buf, c_buf, d_buf }, pc, { n_embd, n_ಟೋಕನುಗಳು, 1 });
}

ಶೂನ್ಯ ggml_vk_mul_mat(ggml_backend_vk_context * ctx, vk_context& subctx, ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_cgraph * cgraph, ಪೂರ್ಣಾಂಕ ಗಂಟು_idx) {
    ggml_tensor * dst = cgraph->nodes[ಗಂಟು_idx];
    ggml_tensor * src0 = dst->src[0];
    ggml_tensor * src1 = dst->src[1];
    VK_LOG_DEBUG("ggml_vk_mul_mat(" << src0 << ", " << src1 << ", " << dst << ")");

    // Handle huge A matrix by splitting the M dimensions. This works well for convolution use cases
    // where the M dimension is very large.
    // Split_k doesn't work with M splitting.
    // This only supports batchsize == 1.
    ಸ್ಥಿರಾಂಕ size_t nbytes = ggml_nbytes(src0);
    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ needs_ವಿಭಜಿಸು = dst->ne[2] == 1 && dst->ne[3] == 1 && nbytes > ctx->device->properties.limits.max_Storage_ಬಫರ್_Range;
    ಆದರೆ (needs_ವಿಭಜಿಸು) {
        // Choose the number of rows that can fit (and divide by two, to allow for any additional offsets)
        ಸ್ಥಿರಾಂಕ uint32_t M_ವಿಭಜಿಸು = ctx->device->properties.limits.max_Storage_ಬಫರ್_Range / (2 * src0->nb[1]);
        uint32_t m_ಸ್ಥಾನಾಂತರ = 0;
        ಆಗ (m_ಸ್ಥಾನಾಂತರ < dst->ne[0]) {
            ಸ್ಥಿರಾಂಕ uint32_t cur_M_ಗಾತ್ರ = std::min(M_ವಿಭಜಿಸು, (uint32_t)(dst->ne[0] - m_ಸ್ಥಾನಾಂತರ));
            ggml_tensor dst2 = *dst;
            ggml_tensor src02 = *src0;

            dst2.view_src = dst->view_src ? dst->view_src : dst;
            src02.view_src = src0->view_src ? src0->view_src : src0;

            dst2.view_offs += m_ಸ್ಥಾನಾಂತರ * dst->nb[0];
            src02.view_offs += m_ಸ್ಥಾನಾಂತರ * src0->nb[1];
            dst2.ne[0] = cur_M_ಗಾತ್ರ;
            src02.ne[1] = cur_M_ಗಾತ್ರ;

            ggml_vk_mul_mat_q_f16(ctx, subctx, &src02, src1, &dst2, ನಿಜ);

            m_ಸ್ಥಾನಾಂತರ += cur_M_ಗಾತ್ರ;
        }
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (ggml_vk_can_use_fwht(ctx, src1, dst)) {
        ggml_vk_fwht(ctx, subctx, src1, dst);
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_F32 && ggml_is_permuted(src0) && ggml_is_permuted(src1) && dst->ne[1] == 1 &&
        // detect 0213 permutation, and batch size of 1
        src0->nb[0] <= src0->nb[2] &&
        src0->nb[2] <= src0->nb[1] &&
        src0->nb[1] <= src0->nb[3] &&
        src1->nb[0] <= src1->nb[2] &&
        src1->nb[2] <= src1->nb[1] &&
        src1->nb[1] <= src1->nb[3] &&
        src0->ne[3] == 1 &&
        src1->ne[3] == 1 &&
        src0->ne[1] <= ctx->device->properties.limits.max_Compute_Work_ಗುಂಪು_ಎಣಿಕೆ[1] &&
        src1->ne[2] <= ctx->device->properties.limits.max_Compute_Work_ಗುಂಪು_ಎಣಿಕೆ[2]) {
        ggml_vk_mul_mat_vec_p021_f16_f32(ctx, subctx, cgraph, ಗಂಟು_idx);
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_F32 && !ggml_is_contiguous(src0) && !ggml_is_transposed(src1) && dst->ne[1] == 1 &&
               !ggml_is_permuted(src0) && !ggml_is_permuted(src1) &&
               src0->ne[3] <= ctx->device->properties.limits.max_Compute_Work_ಗುಂಪು_ಎಣಿಕೆ[0] &&
               src0->ne[1] <= ctx->device->properties.limits.max_Compute_Work_ಗುಂಪು_ಎಣಿಕೆ[1] &&
               src1->ne[2] <= ctx->device->properties.limits.max_Compute_Work_ಗುಂಪು_ಎಣಿಕೆ[2]) {
        ggml_vk_mul_mat_vec_nc_f16_f32(ctx, subctx, cgraph, ಗಂಟು_idx);
    // With one output row, B^T*A has the same flat output as A^T*B.
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (ctx->ಅಂಕೆ_additional_fused_ops == 0 &&
               (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16) &&
               (src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16 || src1->type == GGML_TYPE_BF16 || ggml_is_quantized(src1->type)) &&
               dst->ne[0] == 1 && dst->ne[1] > mul_mat_vec_max_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ &&
               src0->ne[2] == 1 && src0->ne[3] == 1 &&
               src1->ne[2] == 1 && src1->ne[3] == 1 &&
               ggml_is_contiguous(src0) && ggml_is_contiguous(src1) && ggml_is_contiguous(dst) &&
               get_misalign_ಬೈಟುಗಳು(ctx, src0) == 0 && get_misalign_ಬೈಟುಗಳು(ctx, src1) == 0 && get_misalign_ಬೈಟುಗಳು(ctx, dst) == 0) {
        ggml_vk_mul_mat_vec_q_f16(ctx, subctx, cgraph, ಗಂಟು_idx, ನಿಜ);
    // mul_mat_vec supports batching ne12*ne13 when ne11==1, or treating ne11 as the batch size (up to four)
    // when ne12 and ne13 are one.
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ ((dst->ne[1] == 1 || (dst->ne[1] <= mul_mat_vec_max_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ && src1->ne[2] * src1->ne[3] == 1)) &&
               (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_BF16 || ggml_is_quantized(src0->type))) {
        ggml_vk_mul_mat_vec_q_f16(ctx, subctx, cgraph, ಗಂಟು_idx);
    } ಇಲ್ಲದಿದ್ದರೆ {
        ggml_vk_mul_mat_q_f16(ctx, subctx, src0, src1, dst, ಸುಳ್ಳು);
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_vk_mul_mat_id_q_f16(ggml_backend_vk_context * ctx, vk_context& subctx, ಸ್ಥಿರಾಂಕ ggml_tensor * src0, ಸ್ಥಿರಾಂಕ ggml_tensor * src1, ಸ್ಥಿರಾಂಕ ggml_tensor * ids, ggml_tensor * dst) {
    VK_LOG_DEBUG("ggml_vk_mul_mat_id_q_f16((" << src0 << ", name=" << src0->name << ", type=" << src0->type << ", ne0=" << src0->ne[0] << ", ne1=" << src0->ne[1] << ", ne2=" << src0->ne[2] << ", ne3=" << src0->ne[3] << ", nb0=" << src0->nb[0] << ", nb1=" << src0->nb[1] << ", nb2=" << src0->nb[2] << ", nb3=" << src0->nb[3];
    std::cerr << "), (" << src1 << ", name=" << src1->name << ", type=" << src1->type << ", ne0=" << src1->ne[0] << ", ne1=" << src1->ne[1] << ", ne2=" << src1->ne[2] << ", ne3=" << src1->ne[3] << ", nb0=" << src1->nb[0] << ", nb1=" << src1->nb[1] << ", nb2=" << src1->nb[2] << ", nb3=" << src1->nb[3];
    std::cerr << "), (" << ids << ", name=" << ids->name << ", type=" << ids->type << ", ne0=" << ids->ne[0] << ", ne1=" << ids->ne[1] << ", ne2=" << ids->ne[2] << ", ne3=" << ids->ne[3] << ", nb0=" << ids->nb[0] << ", nb1=" << ids->nb[1] << ", nb2=" << ids->nb[2] << ", nb3=" << ids->nb[3];
    std::cerr << "), (" << dst << ", name=" << dst->name << ", type=" << dst->type << ", ne0=" << dst->ne[0] << ", ne1=" << dst->ne[1] << ", ne2=" << dst->ne[2] << ", ne3=" << dst->ne[3] << ", nb0=" << dst->nb[0] << ", nb1=" << dst->nb[1] << ", nb2=" << dst->nb[2] << ", nb3=" << dst->nb[3] << "),)");
    GGML_ASSERT(ggml_vk_dim01_contiguous(src1) || src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16);  // NOLINT
    GGML_ASSERT(ids->type == GGML_TYPE_I32);

    ಸ್ಥಿರಾಂಕ uint64_t ne00 = src0->ne[0];
    ಸ್ಥಿರಾಂಕ uint64_t ne01 = src0->ne[1];
    ಸ್ಥಿರಾಂಕ uint64_t ne02 = src0->ne[2];
    // const uint64_t ne03 = src0->ne[3];

    ಸ್ಥಿರಾಂಕ uint64_t ne10 = src1->ne[0];
    ಸ್ಥಿರಾಂಕ uint64_t ne11 = src1->ne[1];
    ಸ್ಥಿರಾಂಕ uint64_t ne12 = src1->ne[2];
    ಸ್ಥಿರಾಂಕ uint64_t ne13 = src1->ne[3];

    ಸ್ಥಿರಾಂಕ uint64_t nei0 = ids->ne[0];
    ಸ್ಥಿರಾಂಕ uint64_t nei1 = ids->ne[1];

    ಸ್ಥಿರಾಂಕ uint32_t nbi0 = ids->nb[0];
    ಸ್ಥಿರಾಂಕ uint32_t nbi1 = ids->nb[1];

    ಸ್ಥಿರಾಂಕ uint64_t ne20 = dst->ne[0];
    ಸ್ಥಿರಾಂಕ uint64_t ne21 = dst->ne[1];
    // const uint64_t ne22 = dst->ne[2];
    // const uint64_t ne23 = dst->ne[3];

    ಸ್ಥಿರಾಂಕ uint64_t n_as = ne02;

    ggml_backend_vk_ಬಫರ್_context * dst_buf_ctx = (ggml_backend_vk_ಬಫರ್_context *)dst->ಬಫರ್->context;
    ggml_backend_vk_ಬಫರ್_context * src0_buf_ctx = (ggml_backend_vk_ಬಫರ್_context *)src0->ಬಫರ್->context;
    ggml_backend_vk_ಬಫರ್_context * src1_buf_ctx = (ggml_backend_vk_ಬಫರ್_context *)src1->ಬಫರ್->context;

    vk_ಬಫರ್ d_Qx = ಶೂನ್ಯಸೂಚಕ;
    size_t qx_buf_ಸ್ಥಾನಾಂತರ = 0;
    vk_ಬಫರ್ d_Qy = ಶೂನ್ಯಸೂಚಕ;
    size_t qy_buf_ಸ್ಥಾನಾಂತರ = 0;

    ತಾರ್ಕಿಕ src0_uma = ಸುಳ್ಳು;
    ತಾರ್ಕಿಕ src1_uma = ಸುಳ್ಳು;

    ಆದರೆ (ctx->device->uma) {
        ggml_vk_host_get(ctx->device, src0->ದತ್ತಾಂಶ, d_Qx, qx_buf_ಸ್ಥಾನಾಂತರ);
        ggml_vk_host_get(ctx->device, src1->ದತ್ತಾಂಶ, d_Qy, qy_buf_ಸ್ಥಾನಾಂತರ);
        src0_uma = d_Qx != ಶೂನ್ಯಸೂಚಕ;
        src1_uma = d_Qy != ಶೂನ್ಯಸೂಚಕ;
    }

    // Reformat and convert to fp16 if non-contiguous, or for coopmat2 for better perf
    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ x_non_contig = (ctx->device->coopmat2 && src0->type == GGML_TYPE_F32) ||
                              !ggml_vk_dim01_contiguous(src0);
    // If src0 is BF16, try to use a BF16 x BF16 multiply
    ggml_type f16_type = src0->type == GGML_TYPE_BF16 ? GGML_TYPE_BF16 : GGML_TYPE_F16;
#ಆದರೆ defined(GGML_VULKAN_COOPMAT2_DECODE_ಸದಿಶ_GLSLC_SUPPORT)
    // B must already be, or be convertible to, the matmul B type used by this path.
    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ y_decode_ಸದಿಶ_supported = ctx->device->coopmat2_decode_ಸದಿಶ &&
                                           (f16_type != GGML_TYPE_BF16 || ctx->device->coopmat2_bf16_support) &&
                                           (src1->type == GGML_TYPE_F32 || src1->type == f16_type);
    // If B is copied to prealloc_y, we can choose a 4-element-aligned row stride.
    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ y_decode_ಸದಿಶ_ಬಳಸುತ್ತದೆ_prealloc = !ggml_vk_dim01_contiguous(src1) || src1->type != f16_type;
    // Direct B reads are safe only if row starts and the original buffer offset are 4-element aligned.
    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ y_decode_ಸದಿಶ_aligned =
        (ne10 % 4 == 0) &&
        (y_decode_ಸದಿಶ_ಬಳಸುತ್ತದೆ_prealloc || get_misalign_ಬೈಟುಗಳು(ctx, src1) % (4 * ggml_type_ಗಾತ್ರ(src1->type)) == 0);
    // Stage B only when decode-vector is available and direct B reads would be misaligned.
    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ y_decode_ಸದಿಶ_staging = y_decode_ಸದಿಶ_supported && !y_decode_ಸದಿಶ_aligned;
#ಇಲ್ಲದಿದ್ದರೆ
    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ y_decode_ಸದಿಶ_staging = ಸುಳ್ಳು;
#ಆದರೆ_ಅಂತ್ಯ
    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ can_quantize_y = ctx->device->integer_dot_ಗುಣಲಬ್ಧ && src1->type == GGML_TYPE_F32 &&
                                ggml_is_contiguous(src1) && (ne11 * ne10) % 4 == 0;
    // Intel coopmat1 matrix cores prefer f16 activations for quantized weights
    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ prefer_f16_b = ctx->device->coopmat_support && !ctx->device->coopmat2 &&
                              ctx->device->vendor_id == VK_VENDOR_ID_INTEL &&
                              ggml_is_quantized(src0->type) && src1->type == GGML_TYPE_F32;

    ಸ್ಥಿರಾಂಕ ggml_type y_kernel_type = ggml_vk_mul_mat_b_type(ctx, src0->type, src1, f16_type,
                                        (ggml_prec)dst->op_params[0], ನಿಜ, can_quantize_y, prefer_f16_b);

    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ quantize_y   = y_kernel_type == GGML_TYPE_Q8_1;
    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ y_f32_kernel = y_kernel_type == GGML_TYPE_F32;

    ಸ್ಥಿರಾಂಕ std::vector<vk_matmul_pipeline_pair>* mmp_map =
        ggml_vk_get_mul_mat_mat_pipeline_map(ctx, src0->type, y_kernel_type, (ggml_prec)dst->op_params[0], ನಿಜ);

    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ qx_needs_dequant = mmp_map == ಶೂನ್ಯಸೂಚಕ || x_non_contig;

    ಆದರೆ (qx_needs_dequant) {
        // dequant src0 to f16
        mmp_map = ggml_vk_get_mul_mat_mat_pipeline_map(ctx, f16_type, y_kernel_type, (ggml_prec)dst->op_params[0], ನಿಜ);
    }

    GGML_ASSERT(mmp_map != ಶೂನ್ಯಸೂಚಕ);

    // Coopmat2 MUL_MAT_ID BK specialization constants in ggml_vk_load_shaders are at most 64.
    ಸ್ಥಿರಾಂಕ uint32_t y_staged_ಅಡ್ಡಸಾಲು_stride = ctx->device->coopmat2 && !quantize_y ? ggml_vk_align_ಗಾತ್ರ(ne10, 64) : ne10;
    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ y_needs_k_padding = ne10 != y_staged_ಅಡ್ಡಸಾಲು_stride;

    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ y_needs_reformat = !quantize_y && (!ggml_vk_dim01_contiguous(src1) || y_kernel_type != src1->type ||
                                                  y_decode_ಸದಿಶ_staging || y_needs_k_padding);
    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ qy_needs_dequant = y_needs_reformat;

    ಸ್ಥಿರಾಂಕ uint32_t kpad = quantize_y ? 0 : ggml_vk_align_ಗಾತ್ರ(ne10, ggml_vk_guess_matmul_pipeline_align_map(ctx, *mmp_map, ne01, nei1, ನಿಜ));
    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ aligned = !quantize_y && ne10 == kpad && ne01 > 8 && nei1 > 8;

    vk_pipeline pipeline = ggml_vk_guess_matmul_pipeline_map(ctx, *mmp_map, ne01, nei1, aligned, ನಿಜ);

    ಸ್ಥಿರಾಂಕ uint32_t bn = pipeline->wg_denoms[1];
    ಸ್ಥಿರಾಂಕ uint64_t max_tiles = ggml_vk_mul_mat_id_max_tiles(n_as, nei0 * nei1, bn);
    ಸ್ಥಿರಾಂಕ uint64_t ಅಡ್ಡಸಾಲು_ids_ಸ್ಥಾನಾಂತರ = 1 + 2 * max_tiles;
    ಸ್ಥಿರಾಂಕ uint64_t ಅಡ್ಡಸಾಲು_map_ಪದಗಳು = ಅಡ್ಡಸಾಲು_ids_ಸ್ಥಾನಾಂತರ + max_tiles * bn;

    ಆದರೆ (ggml_nbytes(src0) > ctx->device->properties.limits.max_Storage_ಬಫರ್_Range) {
        pipeline = ggml_vk_get_64b_indexing_pipeline(ctx, pipeline);
    }
    ಸ್ಥಿರಾಂಕ uint64_t x_ne = ggml_nelements(src0);
    ಸ್ಥಿರಾಂಕ uint64_t y_ne = (uint64_t)y_staged_ಅಡ್ಡಸಾಲು_stride * ne11 * ne12 * ne13;
    ಸ್ಥಿರಾಂಕ uint64_t d_ne = ggml_nelements(dst);

    ಸ್ಥಿರಾಂಕ uint64_t qx_sz = ggml_type_ಗಾತ್ರ(src0->type) * x_ne / ggml_blck_ಗಾತ್ರ(src0->type);
    ಸ್ಥಿರಾಂಕ uint64_t qy_sz = ggml_type_ಗಾತ್ರ(src1->type) * ggml_nelements(src1) / ggml_blck_ಗಾತ್ರ(src1->type);
    ಸ್ಥಿರಾಂಕ uint64_t x_sz = !qx_needs_dequant ? qx_sz : ಗಾತ್ರ(ggml_fp16_t) * x_ne;
    ಸ್ಥಿರಾಂಕ uint64_t y_sz = quantize_y ? (ggml_vk_align_ಗಾತ್ರ(y_ne, 128) * ggml_type_ಗಾತ್ರ(GGML_TYPE_Q8_1) / ggml_blck_ಗಾತ್ರ(GGML_TYPE_Q8_1)) : (y_f32_kernel ? ಗಾತ್ರ(ದಶಮಾಂಶ) * y_ne : ಗಾತ್ರ(ggml_fp16_t) * y_ne);
    ಸ್ಥಿರಾಂಕ uint64_t d_sz = ಗಾತ್ರ(ದಶಮಾಂಶ) * d_ne;

    vk_pipeline ಗಮ್ಯ_fp16_vk_0 = ಶೂನ್ಯಸೂಚಕ;
    vk_pipeline ಗಮ್ಯ_fp16_vk_1 = ಶೂನ್ಯಸೂಚಕ;
    vk_pipeline ಗಮ್ಯ_q8_1 = ಶೂನ್ಯಸೂಚಕ;

    ಸ್ವಯಂಚಾಲಿತ make_y_staged_dst = [&]() {
        ggml_tensor y_staged_dst = *src1;
        y_staged_dst.type = f16_type;
        y_staged_dst.nb[0] = ggml_type_ಗಾತ್ರ(f16_type);
        y_staged_dst.nb[1] = y_staged_dst.nb[0] * y_staged_ಅಡ್ಡಸಾಲು_stride;
        y_staged_dst.nb[2] = y_staged_dst.nb[1] * ne11;
        y_staged_dst.nb[3] = y_staged_dst.nb[2] * y_staged_dst.ne[2];
        ಹಿಂತಿರುಗಿ y_staged_dst;
    };

    ಆದರೆ (x_non_contig) {
        ಗಮ್ಯ_fp16_vk_0 = ggml_vk_get_cpy_pipeline(ctx, src0, ಶೂನ್ಯಸೂಚಕ, f16_type);
    } ಇಲ್ಲದಿದ್ದರೆ {
        ಗಮ್ಯ_fp16_vk_0 = ggml_vk_get_ಗಮ್ಯ_fp16(ctx, src0->type);
    }
    ಆದರೆ (y_needs_reformat) {
        ggml_tensor y_staged_dst;
        ಸ್ಥಿರಾಂಕ ggml_tensor * y_staged_dst_ptr = ಶೂನ್ಯಸೂಚಕ;
        ಆದರೆ (y_needs_k_padding) {
            y_staged_dst = make_y_staged_dst();
            y_staged_dst_ptr = &y_staged_dst;
        }

        ಗಮ್ಯ_fp16_vk_1 = ggml_vk_get_cpy_pipeline(ctx, src1, y_staged_dst_ptr, f16_type);
    } ಇಲ್ಲದಿದ್ದರೆ {
        ಗಮ್ಯ_fp16_vk_1 = ggml_vk_get_ಗಮ್ಯ_fp16(ctx, src1->type);
    }
    GGML_ASSERT(!qx_needs_dequant || ಗಮ್ಯ_fp16_vk_0 != ಶೂನ್ಯಸೂಚಕ);  // NOLINT
    GGML_ASSERT(!qy_needs_dequant || ಗಮ್ಯ_fp16_vk_1 != ಶೂನ್ಯಸೂಚಕ);  // NOLINT

    ಆದರೆ (quantize_y) {
        ಗಮ್ಯ_q8_1 = ggml_vk_get_quantize_pipeline(ctx, GGML_TYPE_Q8_1);
    }
    vk_pipeline ಎಣಿಕೆ_experts = ctx->device->pipeline_ಎಣಿಕೆ_experts;

    ಸ್ಥಿರಾಂಕ size_t expert_ದತ್ತಾಂಶ_ಗಾತ್ರ = ಗಾತ್ರ(uint32_t) * ಅಡ್ಡಸಾಲು_map_ಪದಗಳು;

    {
        ಆದರೆ (
                (qx_needs_dequant && x_sz > ctx->device->properties.limits.max_Storage_ಬಫರ್_Range) ||
                (qy_needs_dequant && y_sz > ctx->device->properties.limits.max_Storage_ಬಫರ್_Range)) {
            GGML_ABORT("Requested preallocation size is too large");
        }
        ಆದರೆ (qx_needs_dequant && ctx->prealloc_ಗಾತ್ರ_x < x_sz) {
            ctx->prealloc_ಗಾತ್ರ_x = x_sz;
            ggml_vk_preallocate_buffers(ctx, subctx);
        }
        ಆದರೆ ((qy_needs_dequant || quantize_y) && ctx->prealloc_ಗಾತ್ರ_y < y_sz) {
            ctx->prealloc_ಗಾತ್ರ_y = y_sz;
            ggml_vk_preallocate_buffers(ctx, subctx);
        }
        ಆದರೆ (ctx->prealloc_ಗಾತ್ರ_ವಿಭಜಿಸು_k < expert_ದತ್ತಾಂಶ_ಗಾತ್ರ) {
            ctx->prealloc_ಗಾತ್ರ_ವಿಭಜಿಸು_k = expert_ದತ್ತಾಂಶ_ಗಾತ್ರ;
            ggml_vk_preallocate_buffers(ctx, subctx);
        }

        // Request descriptor sets
        ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);
        ಆದರೆ (qx_needs_dequant) {
            ggml_pipeline_request_descriptor_sets(ctx, ಗಮ್ಯ_fp16_vk_0, 1);
        }
        ಆದರೆ (qy_needs_dequant) {
            ggml_pipeline_request_descriptor_sets(ctx, ಗಮ್ಯ_fp16_vk_1, 1);
        }
        ಆದರೆ (quantize_y) {
            ggml_pipeline_request_descriptor_sets(ctx, ಗಮ್ಯ_q8_1, 1);
        }
        ggml_pipeline_request_descriptor_sets(ctx, ಎಣಿಕೆ_experts, 1);
    }

    vk_ಬಫರ್ d_D = dst_buf_ctx->dev_ಬಫರ್;
    ಸ್ಥಿರಾಂಕ uint64_t d_buf_ಸ್ಥಾನಾಂತರ = vk_tensor_ಸ್ಥಾನಾಂತರ(dst) + dst->view_offs;
    GGML_ASSERT(d_D != ಶೂನ್ಯಸೂಚಕ);
    vk_ಬಫರ್ d_X;
    uint64_t x_buf_ಸ್ಥಾನಾಂತರ = 0;
    vk_ಬಫರ್ d_Y;
    uint64_t y_buf_ಸ್ಥಾನಾಂತರ = 0;
    ಆದರೆ (!src0_uma) {
        d_Qx = src0_buf_ctx->dev_ಬಫರ್;
        qx_buf_ಸ್ಥಾನಾಂತರ = vk_tensor_ಸ್ಥಾನಾಂತರ(src0) + src0->view_offs;
        GGML_ASSERT(d_Qx != ಶೂನ್ಯಸೂಚಕ);
    }
    ಆದರೆ (!src1_uma) {
        d_Qy = src1_buf_ctx->dev_ಬಫರ್;
        qy_buf_ಸ್ಥಾನಾಂತರ = vk_tensor_ಸ್ಥಾನಾಂತರ(src1) + src1->view_offs;
        GGML_ASSERT(d_Qy != ಶೂನ್ಯಸೂಚಕ);
    }
    ಆದರೆ (qx_needs_dequant) {
        d_X = ctx->prealloc_x;
        GGML_ASSERT(d_X->size >= x_sz);
    } ಇಲ್ಲದಿದ್ದರೆ {
        d_X = d_Qx;
        x_buf_ಸ್ಥಾನಾಂತರ = qx_buf_ಸ್ಥಾನಾಂತರ;
        GGML_ASSERT(qx_sz == x_sz);
    }
    ಆದರೆ (qy_needs_dequant) {
        d_Y = ctx->prealloc_y;
        GGML_ASSERT(d_Y->size >= y_sz);
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (quantize_y) {
        d_Y = ctx->prealloc_y;
        GGML_ASSERT(d_Y->size >= CEIL_DIV(y_sz, 144) * 144);
    } ಇಲ್ಲದಿದ್ದರೆ {
        d_Y = d_Qy;
        y_buf_ಸ್ಥಾನಾಂತರ = qy_buf_ಸ್ಥಾನಾಂತರ;
        GGML_ASSERT(qy_sz == y_sz);
    }

    ಆದರೆ (x_non_contig || qx_needs_dequant) {
        ಆದರೆ (ctx->prealloc_x_need_sync) {
            ggml_vk_sync_buffers(ctx, subctx);
        }
    }
    vk_subbuffer d_ids = ggml_vk_tensor_subbuffer(ctx, ids, ನಿಜ);

    // Count how many times each expert is used
    vk_subbuffer expert_ಎಣಿಕೆ_buf = { ctx->prealloc_ವಿಭಜಿಸು_k, 0, expert_ದತ್ತಾಂಶ_ಗಾತ್ರ };
    ಆದರೆ (ctx->prealloc_ವಿಭಜಿಸು_k_need_sync) {
        ggml_vk_sync_buffers(ctx, subctx);
    }
    {
        vk_op_ಎಣಿಕೆ_experts_ತಳ್ಳು_constants pc = { (uint32_t)nei0,
                                           (uint32_t)nei1,
                                           (uint32_t)(nbi0 / ggml_type_ಗಾತ್ರ(ids->type)),
                                           (uint32_t)(nbi1 / ggml_type_ಗಾತ್ರ(ids->type)),
                                           (uint32_t)(get_misalign_ಬೈಟುಗಳು(ctx, ids) / ggml_type_ಗಾತ್ರ(ids->type)),
                                           (uint32_t)n_as,
                                           0, 0, bn, (uint32_t)ಅಡ್ಡಸಾಲು_ids_ಸ್ಥಾನಾಂತರ };
        init_pushconst_fastdiv(pc);
        ggml_vk_dispatch_pipeline(ctx, subctx, ಎಣಿಕೆ_experts,
            { d_ids, expert_ಎಣಿಕೆ_buf }, pc,
            { 1, 1, 1 });
    }

    ಆದರೆ (x_non_contig) {
        ggml_vk_cpy_ಗಮ್ಯ_contiguous(ctx, subctx, ಗಮ್ಯ_fp16_vk_0, src0, ggml_vk_subbuffer(ctx, d_Qx, qx_buf_ಸ್ಥಾನಾಂತರ), ggml_vk_subbuffer(ctx, d_X, 0));
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (qx_needs_dequant) {
        ಸ್ಥಿರಾಂಕ std::vector<uint32_t> pc = { (uint32_t)ne01, (uint32_t)ne10, (uint32_t)ne10, (uint32_t)ne10, (uint32_t)(ggml_nelements(src0)) };
        ggml_vk_dispatch_pipeline(ctx, subctx, ಗಮ್ಯ_fp16_vk_0,
            { vk_subbuffer{ d_Qx, qx_buf_ಸ್ಥಾನಾಂತರ, qx_sz }, vk_subbuffer{ d_X, 0, x_sz } }, pc, { (uint32_t)x_ne, 1, 1});
    }
    ಆದರೆ (y_needs_reformat) {
        ಆದರೆ (ctx->prealloc_y_ಕೊನೆಯ_pipeline_ಬಳಸಿದ != ಗಮ್ಯ_fp16_vk_1.get() ||
            ctx->prealloc_y_ಕೊನೆಯ_tensor_ಬಳಸಿದ != src1 ||
            ctx->prealloc_y_ಕೊನೆಯ_k_padded != y_needs_k_padding) {
            ಆದರೆ (ctx->prealloc_y_need_sync) {
                ggml_vk_sync_buffers(ctx, subctx);
            }
            ಆದರೆ (y_needs_k_padding) {
                GGML_ASSERT(y_sz % 4 == 0);
                // Zero B padding because clamping only A can produce 0 * Inf or NaN.
                subctx->s->ಬಫರ್->buf.fill_ಬಫರ್(d_Y->ಬಫರ್, 0, y_sz, 0);
                ggml_vk_sync_buffers(ctx, subctx);
                ಸ್ಥಿರಾಂಕ ggml_tensor y_staged_dst = make_y_staged_dst();
                ಸ್ಥಿರಾಂಕ uint32_t y_staged_dst_type_ಗಾತ್ರ = ggml_type_ಗಾತ್ರ(y_staged_dst.type);
                ggml_vk_cpy_ಗಮ್ಯ_strided(
                    ctx, subctx, ಗಮ್ಯ_fp16_vk_1, src1,
                    ggml_vk_subbuffer(ctx, d_Qy, qy_buf_ಸ್ಥಾನಾಂತರ), ggml_vk_subbuffer(ctx, d_Y, 0),
                    (uint32_t)(y_staged_dst.nb[0] / y_staged_dst_type_ಗಾತ್ರ),
                    (uint32_t)(y_staged_dst.nb[1] / y_staged_dst_type_ಗಾತ್ರ),
                    (uint32_t)(y_staged_dst.nb[2] / y_staged_dst_type_ಗಾತ್ರ),
                    (uint32_t)(y_staged_dst.nb[3] / y_staged_dst_type_ಗಾತ್ರ));
            } ಇಲ್ಲದಿದ್ದರೆ {
                ggml_vk_cpy_ಗಮ್ಯ_contiguous(ctx, subctx, ಗಮ್ಯ_fp16_vk_1, src1, ggml_vk_subbuffer(ctx, d_Qy, qy_buf_ಸ್ಥಾನಾಂತರ), ggml_vk_subbuffer(ctx, d_Y, 0));
            }
            ctx->prealloc_y_ಕೊನೆಯ_pipeline_ಬಳಸಿದ = ಗಮ್ಯ_fp16_vk_1.get();
            ctx->prealloc_y_ಕೊನೆಯ_tensor_ಬಳಸಿದ = src1;
            ctx->prealloc_y_ಕೊನೆಯ_k_padded = y_needs_k_padding;
        }
    }
    ಆದರೆ (quantize_y) {
        ಆದರೆ (ctx->prealloc_y_ಕೊನೆಯ_pipeline_ಬಳಸಿದ != ಗಮ್ಯ_q8_1.get() ||
            ctx->prealloc_y_ಕೊನೆಯ_tensor_ಬಳಸಿದ != src1 ||
            ctx->prealloc_y_ಕೊನೆಯ_k_padded) {
            ಆದರೆ (ctx->prealloc_y_need_sync) {
                ggml_vk_sync_buffers(ctx, subctx);
            }
            ggml_vk_quantize_q8_1(ctx, subctx, ggml_vk_subbuffer(ctx, d_Qy, qy_buf_ಸ್ಥಾನಾಂತರ), ggml_vk_subbuffer(ctx, d_Y, 0), y_ne);
            ctx->prealloc_y_ಕೊನೆಯ_pipeline_ಬಳಸಿದ = ಗಮ್ಯ_q8_1.get();
            ctx->prealloc_y_ಕೊನೆಯ_tensor_ಬಳಸಿದ = src1;
            ctx->prealloc_y_ಕೊನೆಯ_k_padded = ಸುಳ್ಳು;
        }
    }
    ggml_vk_sync_buffers(ctx, subctx);

    ಸ್ಥಿರಾಂಕ uint64_t x_range = qx_needs_dequant ? x_sz : ggml_nbytes(src0);
    ಸ್ಥಿರಾಂಕ uint64_t y_range = (qy_needs_dequant || quantize_y) ? y_sz : ggml_nbytes(src1);

    uint32_t stride_batch_x = qx_needs_dequant ? ne00*ne01 : ggml_vk_batch_stride(src0);
    uint32_t stride_b_y = y_needs_k_padding ? y_staged_ಅಡ್ಡಸಾಲು_stride : ne10;
    uint32_t stride_batch_y = y_needs_k_padding ? y_staged_ಅಡ್ಡಸಾಲು_stride * ne11 : ((qy_needs_dequant || quantize_y) ? ne10*ne11 : ggml_vk_batch_stride(src1));

    ಆದರೆ (!ggml_vk_dim01_contiguous(src0) && !qx_needs_dequant) {
        stride_batch_x = src0->nb[0] / ggml_type_ಗಾತ್ರ(src0->type);
    }

    ಆದರೆ (!ggml_vk_dim01_contiguous(src1) && !qy_needs_dequant && !quantize_y) {
        stride_batch_y = src1->nb[0] / ggml_type_ಗಾತ್ರ(src1->type);
    }

    // compute
    ggml_vk_matmul_id(
        ctx, subctx, pipeline,
        { d_X, x_buf_ಸ್ಥಾನಾಂತರ, x_range }, { d_Y, y_buf_ಸ್ಥಾನಾಂತರ, y_range },
        { d_D, d_buf_ಸ್ಥಾನಾಂತರ, d_sz }, std::move(d_ids), expert_ಎಣಿಕೆ_buf,
        ne01, ne21, ne10, ne10, stride_b_y, ne01,
        stride_batch_x, stride_batch_y, ne20*ne21,
        n_as, nei0, nei1, nbi1 / ggml_type_ಗಾತ್ರ(ids->type), ne11, ಅಡ್ಡಸಾಲು_ids_ಸ್ಥಾನಾಂತರ, max_tiles
    );  // NOLINT

    ಆದರೆ (x_non_contig || qx_needs_dequant) {
        ctx->prealloc_x_need_sync = ನಿಜ;
    }
    ಆದರೆ (y_needs_reformat || quantize_y) {
        ctx->prealloc_y_need_sync = ನಿಜ;
    }
    ctx->prealloc_ವಿಭಜಿಸು_k_need_sync = ನಿಜ;
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_vk_mul_mat_vec_id_q_f16(ggml_backend_vk_context * ctx, vk_context& subctx, ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_cgraph * cgraph, ಪೂರ್ಣಾಂಕ ಗಂಟು_idx) {
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

    ಸ್ಥಿರಾಂಕ uint64_t ne00 = src0->ne[0];
    ಸ್ಥಿರಾಂಕ uint64_t ne01 = src0->ne[1];
    // const uint64_t ne02 = src0->ne[2];
    // const uint64_t ne03 = src0->ne[3];

    ಸ್ಥಿರಾಂಕ uint64_t ne10 = src1->ne[0];
    ಸ್ಥಿರಾಂಕ uint64_t ne11 = src1->ne[1];
    ಸ್ಥಿರಾಂಕ uint64_t ne12 = src1->ne[2];
    // const uint64_t ne13 = src1->ne[3];

    ಸ್ಥಿರಾಂಕ uint64_t nei0 = ids->ne[0];
    ಸ್ಥಿರಾಂಕ uint64_t nei1 = ids->ne[1];
    ಸ್ಥಿರಾಂಕ uint32_t nbi1 = (uint32_t)(ids->nb[1] / ಗಾತ್ರ(ಪೂರ್ಣಾಂಕ));

    ಸ್ಥಿರಾಂಕ uint64_t ne20 = dst->ne[0];
    ಸ್ಥಿರಾಂಕ uint64_t ne21 = dst->ne[1];
    // const uint64_t ne22 = dst->ne[2];
    // const uint64_t ne23 = dst->ne[3];

    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ x_non_contig = !ggml_vk_dim01_contiguous(src0);
    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ y_non_contig = !ggml_vk_dim01_contiguous(src1);

    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ f16_f32_kernel = src1->type == GGML_TYPE_F32;
    ತಾರ್ಕಿಕ quantize_y = ctx->device->integer_dot_ಗುಣಲಬ್ಧ && src1->type == GGML_TYPE_F32 && ggml_is_contiguous(src1) && !y_non_contig && (ne11 * ne10) % 4 == 0 && ggml_vk_should_use_mmvq(ctx->device, ne01, ne12, ne10, src0->type);

    vk_pipeline ಗಮ್ಯ_fp16_vk_0 = ಶೂನ್ಯಸೂಚಕ;
    vk_pipeline ಗಮ್ಯ_fp16_vk_1 = ಶೂನ್ಯಸೂಚಕ;
    ಆದರೆ (x_non_contig) {
        ಗಮ್ಯ_fp16_vk_0 = ggml_vk_get_cpy_pipeline(ctx, src0, ಶೂನ್ಯಸೂಚಕ, src0->type);
    }
    ಆದರೆ (y_non_contig) {
        ಗಮ್ಯ_fp16_vk_1 = ggml_vk_get_cpy_pipeline(ctx, src1, ಶೂನ್ಯಸೂಚಕ, src1->type);
    } ಇಲ್ಲದಿದ್ದರೆ {
        ಗಮ್ಯ_fp16_vk_1 = ggml_vk_get_ಗಮ್ಯ_fp16(ctx, src1->type);
    }

    // Check for mmq first
    vk_pipeline dmmv = quantize_y ? ggml_vk_get_dequantize_mul_mat_vec_id(ctx, src0->type, GGML_TYPE_Q8_1, ne20, ne00) : ಶೂನ್ಯಸೂಚಕ;
    vk_pipeline ಗಮ್ಯ_q8_1 = ಶೂನ್ಯಸೂಚಕ;

    ಆದರೆ (dmmv == ಶೂನ್ಯಸೂಚಕ) {
        // Fall back to f16 dequant mul mat
        dmmv = ggml_vk_get_dequantize_mul_mat_vec_id(ctx, src0->type, src1->type, ne20, ne00);
        quantize_y = ಸುಳ್ಳು;
    }

    ಆದರೆ (quantize_y) {
        ಗಮ್ಯ_q8_1 = ggml_vk_get_quantize_pipeline(ctx, GGML_TYPE_Q8_1);
    }

    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ qx_needs_dequant = x_non_contig;
    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ qy_needs_dequant = !quantize_y && ((src1->type != GGML_TYPE_F16 && !f16_f32_kernel) || y_non_contig);

    ಆದರೆ (ggml_nbytes(src0) > ctx->device->properties.limits.max_Storage_ಬಫರ್_Range) {
        dmmv = ggml_vk_get_64b_indexing_pipeline(ctx, dmmv);
    }

    // Not implemented
    GGML_ASSERT(y_non_contig || !qy_needs_dequant);  // NOLINT
    GGML_ASSERT(!qx_needs_dequant || ಗಮ್ಯ_fp16_vk_0 != ಶೂನ್ಯಸೂಚಕ);  // NOLINT
    GGML_ASSERT(!qy_needs_dequant || ಗಮ್ಯ_fp16_vk_1 != ಶೂನ್ಯಸೂಚಕ);  // NOLINT
    GGML_ASSERT(dmmv != ಶೂನ್ಯಸೂಚಕ);

    ಸ್ಥಿರಾಂಕ uint64_t x_ne = ggml_nelements(src0);
    ಸ್ಥಿರಾಂಕ uint64_t y_ne = ggml_nelements(src1);

    ಸ್ಥಿರಾಂಕ uint64_t qx_sz = ggml_vk_align_ಗಾತ್ರ(ggml_type_ಗಾತ್ರ(src0->type) * x_ne / ggml_blck_ಗಾತ್ರ(src0->type), ctx->device->properties.limits.min_Storage_ಬಫರ್_ಸ್ಥಾನಾಂತರ_Alignment);
    ಸ್ಥಿರಾಂಕ uint64_t x_sz = x_non_contig ? ggml_vk_align_ಗಾತ್ರ(ggml_type_ಗಾತ್ರ(src0->type) * x_ne, ctx->device->properties.limits.min_Storage_ಬಫರ್_ಸ್ಥಾನಾಂತರ_Alignment) : qx_sz;
    ಸ್ಥಿರಾಂಕ uint64_t y_sz = quantize_y ? (ggml_vk_align_ಗಾತ್ರ(y_ne, 128) * ggml_type_ಗಾತ್ರ(GGML_TYPE_Q8_1) / ggml_blck_ಗಾತ್ರ(GGML_TYPE_Q8_1)) :
                                       (f16_f32_kernel ? ಗಾತ್ರ(ದಶಮಾಂಶ) * y_ne : ಗಾತ್ರ(ggml_fp16_t) * y_ne);

    {
        ಆದರೆ (
                (qx_needs_dequant && x_sz > ctx->device->properties.limits.max_Storage_ಬಫರ್_Range) ||
                (qy_needs_dequant && y_sz > ctx->device->properties.limits.max_Storage_ಬಫರ್_Range)) {
            GGML_ABORT("Requested preallocation size is too large");
        }
        ಆದರೆ (qx_needs_dequant && ctx->prealloc_ಗಾತ್ರ_x < x_sz) {
            ctx->prealloc_ಗಾತ್ರ_x = x_sz;
            ggml_vk_preallocate_buffers(ctx, subctx);
        }
        ಆದರೆ ((qy_needs_dequant || quantize_y) && ctx->prealloc_ಗಾತ್ರ_y < y_sz) {
            ctx->prealloc_ಗಾತ್ರ_y = y_sz;
            ggml_vk_preallocate_buffers(ctx, subctx);
        }

        // Request descriptor sets
        ಆದರೆ (qx_needs_dequant) {
            ggml_pipeline_request_descriptor_sets(ctx, ಗಮ್ಯ_fp16_vk_0, 1);
        }
        ಆದರೆ (qy_needs_dequant) {
            ggml_pipeline_request_descriptor_sets(ctx, ಗಮ್ಯ_fp16_vk_1, 1);
        }
        ಆದರೆ (quantize_y) {
            ggml_pipeline_request_descriptor_sets(ctx, ಗಮ್ಯ_q8_1, 1);
        }
        ggml_pipeline_request_descriptor_sets(ctx, dmmv, nei1);
    }

    vk_subbuffer d_D = ggml_vk_tensor_subbuffer(ctx, cgraph->nodes[ಗಂಟು_idx + ctx->ಅಂಕೆ_additional_fused_ops]);
    vk_subbuffer d_Qx = ggml_vk_tensor_subbuffer(ctx, src0);
    vk_subbuffer d_Qy = ggml_vk_tensor_subbuffer(ctx, src1);
    vk_subbuffer d_ids = ggml_vk_tensor_subbuffer(ctx, ids);
    vk_subbuffer d_F0 = d_D;
    vk_subbuffer d_X, d_Y;

    ಆದರೆ (qx_needs_dequant) {
        d_X = { ctx->prealloc_x, 0, ctx->prealloc_x->size };
    } ಇಲ್ಲದಿದ್ದರೆ {
        d_X = d_Qx;
    }
    ಆದರೆ (qy_needs_dequant || quantize_y) {
        d_Y = { ctx->prealloc_y, 0, ctx->prealloc_y->size };
    } ಇಲ್ಲದಿದ್ದರೆ {
        d_Y = d_Qy;
    }

    ಆದರೆ (x_non_contig) {
        ಆದರೆ (ctx->prealloc_x_need_sync) {
            ggml_vk_sync_buffers(ctx, subctx);
        }
    }

    ಆದರೆ (x_non_contig) {
        GGML_ASSERT(x_sz == ggml_vk_align_ಗಾತ್ರ(ggml_type_ಗಾತ್ರ(src0->type) * x_ne, ctx->device->properties.limits.min_Storage_ಬಫರ್_ಸ್ಥಾನಾಂತರ_Alignment));
        ggml_vk_cpy_ಗಮ್ಯ_contiguous(ctx, subctx, ಗಮ್ಯ_fp16_vk_0, src0, d_Qx, d_X);
    }
    ಆದರೆ (y_non_contig) {
        GGML_ASSERT(y_sz == ggml_type_ಗಾತ್ರ(src1->type) * y_ne);
        ಆದರೆ (ctx->prealloc_y_ಕೊನೆಯ_pipeline_ಬಳಸಿದ != ಗಮ್ಯ_fp16_vk_1.get() ||
            ctx->prealloc_y_ಕೊನೆಯ_tensor_ಬಳಸಿದ != src1 ||
            ctx->prealloc_y_ಕೊನೆಯ_k_padded) {
            ಆದರೆ (ctx->prealloc_y_need_sync) {
                ggml_vk_sync_buffers(ctx, subctx);
            }
            ggml_vk_cpy_ಗಮ್ಯ_contiguous(ctx, subctx, ಗಮ್ಯ_fp16_vk_1, src1, d_Qy, d_Y);
            ctx->prealloc_y_ಕೊನೆಯ_pipeline_ಬಳಸಿದ = ಗಮ್ಯ_fp16_vk_1.get();
            ctx->prealloc_y_ಕೊನೆಯ_tensor_ಬಳಸಿದ = src1;
            ctx->prealloc_y_ಕೊನೆಯ_k_padded = ಸುಳ್ಳು;
        }
    }
    ಆದರೆ (quantize_y) {
        ಆದರೆ (ctx->prealloc_y_ಕೊನೆಯ_pipeline_ಬಳಸಿದ != ಗಮ್ಯ_q8_1.get() ||
            ctx->prealloc_y_ಕೊನೆಯ_tensor_ಬಳಸಿದ != src1 ||
            ctx->prealloc_y_ಕೊನೆಯ_k_padded) {
            ಆದರೆ (ctx->prealloc_y_need_sync) {
                ggml_vk_sync_buffers(ctx, subctx);
            }
            ggml_vk_quantize_q8_1(ctx, subctx, d_Qy, d_Y, y_ne);
            ctx->prealloc_y_ಕೊನೆಯ_pipeline_ಬಳಸಿದ = ಗಮ್ಯ_q8_1.get();
            ctx->prealloc_y_ಕೊನೆಯ_tensor_ಬಳಸಿದ = src1;
            ctx->prealloc_y_ಕೊನೆಯ_k_padded = ಸುಳ್ಳು;
        }
    }

    uint32_t stride_batch_x = qx_needs_dequant ? ne00*ne01 : ggml_vk_batch_stride(src0);
    uint32_t stride_batch_y = (qy_needs_dequant || quantize_y) ? ne10*ne11 : ggml_vk_batch_stride(src1);

    ಆದರೆ (!ggml_vk_dim01_contiguous(src1) && !qy_needs_dequant) {
        stride_batch_y = src1->nb[2] / ggml_type_ಗಾತ್ರ(src1->type);
    }

    ಸ್ಥಿರಾಂಕ uint32_t max_groups_x = ctx->device->properties.limits.max_Compute_Work_ಗುಂಪು_ಎಣಿಕೆ[0];

    uint32_t groups_x = ne01;
    uint32_t groups_z = 1;

    ಆದರೆ (ne01 > max_groups_x) {
        groups_z = 64;
        groups_x = CEIL_DIV(groups_x, groups_z);
    }

    uint32_t fusion_flags = 0;

    ಆದರೆ (ctx->ಅಂಕೆ_additional_fused_ops > 0) {
        ಸ್ಥಿರಾಂಕ ggml_tensor * bias = cgraph->nodes[ಗಂಟು_idx + 1]->src[1];

        d_F0 = ggml_vk_tensor_subbuffer(ctx, bias);

        ಆದರೆ (cgraph->nodes[ಗಂಟು_idx + 1]->op == GGML_OP_MUL) {
            fusion_flags |= MAT_VEC_FUSION_FLAGS_SCALE0;
        } ಇಲ್ಲದಿದ್ದರೆ {
            GGML_ASSERT(cgraph->nodes[ಗಂಟು_idx + 1]->op == GGML_OP_ಕೂಡಿಸು_ID);
            fusion_flags |= MAT_VEC_FUSION_FLAGS_BIAS0;
        }
    }

    vk_subbuffer d_F1 = d_D;
    ಆದರೆ (ctx->ಅಂಕೆ_additional_fused_ops > 1) {
        ಸ್ಥಿರಾಂಕ ggml_tensor * scale = cgraph->nodes[ಗಂಟು_idx + 2]->src[1];

        d_F1 = ggml_vk_tensor_subbuffer(ctx, scale);
        fusion_flags |= MAT_VEC_FUSION_FLAGS_SCALE1;
    }

    // Loop over the batch dimension
    ಗಾಗಿ (uint32_t expert_i1 = 0; expert_i1 < nei1; ++expert_i1) {
        ಸ್ಥಿರಾಂಕ vk_mat_vec_id_ತಳ್ಳು_constants pc = {
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

    ಆದರೆ (x_non_contig) {
        ctx->prealloc_x_need_sync = ನಿಜ;
    }
    ಆದರೆ (y_non_contig || quantize_y) {
        ctx->prealloc_y_need_sync = ನಿಜ;
    }
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_vk_use_mul_mat_vec_id(ಸ್ಥಿರಾಂಕ ggml_tensor * dst) {
    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = dst->src[0];
    ಸ್ಥಿರಾಂಕ ggml_tensor * ids = dst->src[2];
    ಹಿಂತಿರುಗಿ ids->ne[1] <= 8 && (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16 || ggml_is_quantized(src0->type));
}

ತಾರ್ಕಿಕ ggml_vk_use_mul_mat_vec_id(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_cgraph * cgraph, ಪೂರ್ಣಾಂಕ ಗಂಟು_idx) {
    ಹಿಂತಿರುಗಿ ggml_vk_use_mul_mat_vec_id(cgraph->nodes[ಗಂಟು_idx]);
}

ಶೂನ್ಯ ggml_vk_mul_mat_id(ggml_backend_vk_context * ctx, vk_context& subctx, ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_cgraph * cgraph, ಪೂರ್ಣಾಂಕ ಗಂಟು_idx) {
    ggml_tensor * dst = cgraph->nodes[ಗಂಟು_idx];
    ggml_tensor * src0 = dst->src[0];
    ggml_tensor * src1 = dst->src[1];
    ggml_tensor * src2 = dst->src[2];
    VK_LOG_DEBUG("ggml_vk_mul_mat_id(" << src0 << ", " << src1 << ", " << src2 << ", " << dst << ")");
    ಆದರೆ (ggml_vk_use_mul_mat_vec_id(cgraph, ಗಂಟು_idx)) {
        ggml_vk_mul_mat_vec_id_q_f16(ctx, subctx, cgraph, ಗಂಟು_idx);
    } ಇಲ್ಲದಿದ್ದರೆ {
        ggml_vk_mul_mat_id_q_f16(ctx, subctx, src0, src1, src2, dst);
    }
}

ತಾರ್ಕಿಕ ggml_vk_flash_attn_scalar_shmem_support(ಸ್ಥಿರಾಂಕ vk_device& device, ಸ್ಥಿರಾಂಕ vk_fa_tuning_params& params, uint32_t hsk, uint32_t hsv, ತಾರ್ಕಿಕ f32acc, ggml_type k_type, ggml_type v_type) {
    GGML_ಬಳಸದ(f32acc);
    // Needs to be kept up to date on shader changes
    ಸ್ಥಿರಾಂಕ uint32_t wg_ಗಾತ್ರ = params.workgroup_ಗಾತ್ರ;
    ಸ್ಥಿರಾಂಕ uint32_t Br = params.ಖಂಡ_ಅಡ್ಡಸಾಲುಗಳು;
    ಸ್ಥಿರಾಂಕ uint32_t Bc = params.ಖಂಡ_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ;

    // BF16 uses the fp32 shader (FLOAT_TYPE=float)
    ಸ್ಥಿರಾಂಕ uint32_t float_type_ಗಾತ್ರ = (device->fp16 && k_type != GGML_TYPE_BF16) ? ಗಾತ್ರ(ggml_fp16_t) : ಗಾತ್ರ(ದಶಮಾಂಶ);

    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ mmq = ggml_vk_fa_scalar_ಬಳಸುತ್ತದೆ_mmq(device, k_type, v_type);

    // tmpsh is overestimated slightly
    ಸ್ಥಿರಾಂಕ uint32_t tmpsh = wg_ಗಾತ್ರ * ಗಾತ್ರ(ದಶಮಾಂಶ);
    ಸ್ಥಿರಾಂಕ uint32_t tmpshv4 = wg_ಗಾತ್ರ * 4 * float_type_ಗಾತ್ರ;

    ಸ್ಥಿರಾಂಕ uint32_t masksh = Bc * (Br + 1) * float_type_ಗಾತ್ರ;
    // DATA_A_IQ4_NL is compiled into the FA shaders unconditionally, so its shared table is always allocated.
    ಸ್ಥಿರಾಂಕ uint32_t iq_shmem = 16 * float_type_ಗಾತ್ರ;

    uint32_t Qf, kvsh, kblocksh_ಗಾತ್ರ;
    ಆದರೆ (mmq) {
        // block_b_cache: int32_t qs[8] + FLOAT_TYPEV2 ds
        ಸ್ಥಿರಾಂಕ uint32_t ಖಂಡ_b_ಗಾತ್ರ = 8 * ಗಾತ್ರ(int32_t) + 2 * float_type_ಗಾತ್ರ;
        Qf = Br * (hsk / 32) * ಖಂಡ_b_ಗಾತ್ರ;

        // kvsh uses D = HSV (K goes through kblocksh instead)
        kvsh = params.shmem_staging ? Bc * (hsv / 4 + 1) * 4 * float_type_ಗಾತ್ರ : 4 * float_type_ಗಾತ್ರ;

        // The mixed MMQ shader uses a superset block_a_cache that fits every
        // FA-supported quant: int32_t qs[8] + uint32_t qh + FLOAT_TYPEV2 dm.
        // Single-scale types leave dm.y unused; non-Q5_* leave qh unused.
        ಸ್ಥಿರಾಂಕ uint32_t ಖಂಡ_a_ಗಾತ್ರ = 8 * ಗಾತ್ರ(int32_t) + ಗಾತ್ರ(uint32_t) + 2 * float_type_ಗಾತ್ರ;
        kblocksh_ಗಾತ್ರ = params.shmem_staging ? Bc * (hsk / 32) * ಖಂಡ_a_ಗಾತ್ರ : ಖಂಡ_a_ಗಾತ್ರ;
    } ಇಲ್ಲದಿದ್ದರೆ {
        Qf = Br * (hsk / 4 + 1) * 4 * float_type_ಗಾತ್ರ;

        ಸ್ಥಿರಾಂಕ uint32_t D = std::max(hsk, hsv);
        kvsh = params.shmem_staging ? Bc * (D / 4 + 1) * 4 * float_type_ಗಾತ್ರ : 4 * float_type_ಗಾತ್ರ;

        kblocksh_ಗಾತ್ರ = 0;
    }

    ಸ್ಥಿರಾಂಕ uint32_t ಒಟ್ಟು_ಗಾತ್ರ = tmpsh + tmpshv4 + masksh + iq_shmem + Qf + kvsh + kblocksh_ಗಾತ್ರ;
    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ supported = ಒಟ್ಟು_ಗಾತ್ರ <= device->properties.limits.max_Compute_Shared_ಸ್ಮರಣೆ_ಗಾತ್ರ;

    VK_LOG_DEBUG("ggml_vk_flash_attn_scalar_shmem_support(HSK=" << hsk << ", HSV=" << hsv << ", mmq=" << mmq << ", total_size=" << ಒಟ್ಟು_ಗಾತ್ರ << ", supported=" << supported);

    ಹಿಂತಿರುಗಿ supported;
}

ತಾರ್ಕಿಕ ggml_vk_flash_attn_coopmat_shmem_support(ಸ್ಥಿರಾಂಕ vk_device& device, ಸ್ಥಿರಾಂಕ vk_fa_tuning_params& params, uint32_t hsk, uint32_t hsv, ತಾರ್ಕಿಕ f32acc, ggml_type k_type, ggml_type v_type) {
    GGML_ಬಳಸದ(v_type);
    // Needs to be kept up to date on shader changes
    ಸ್ಥಿರಾಂಕ uint32_t Br = params.ಖಂಡ_ಅಡ್ಡಸಾಲುಗಳು;
    ಸ್ಥಿರಾಂಕ uint32_t Bc = params.ಖಂಡ_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ;

    ಸ್ಥಿರಾಂಕ uint32_t MatBr = 16, MatBc = 16;

    ಸ್ಥಿರಾಂಕ uint32_t ಅಡ್ಡಸಾಲು_ವಿಭಜಿಸು = Bc / MatBc;

    ಸ್ಥಿರಾಂಕ uint32_t hsk_pad = ROUNDUP_POW2(hsk, 16);
    ಸ್ಥಿರಾಂಕ uint32_t hsv_pad = ROUNDUP_POW2(hsv, 16);

    ಸ್ಥಿರಾಂಕ uint32_t acctype = f32acc ? 4 : 2;
    ಸ್ಥಿರಾಂಕ uint32_t f16vec4 = 8;

    ಸ್ಥಿರಾಂಕ uint32_t tmpsh = (Bc / MatBc) * ಗಾತ್ರ(ದಶಮಾಂಶ);
    // DATA_A_IQ4_NL is compiled into the FA shaders unconditionally, so its shared table is always allocated.
    ಸ್ಥಿರಾಂಕ uint32_t iq_shmem = 16 * ಗಾತ್ರ(ggml_fp16_t);

    ಸ್ಥಿರಾಂಕ uint32_t qstride = hsk_pad / 4 + 2;
    ಸ್ಥಿರಾಂಕ uint32_t Qf = Br * qstride * f16vec4;

    ಸ್ಥಿರಾಂಕ uint32_t psh_stride = Br / 4 + 2;
    ಸ್ಥಿರಾಂಕ uint32_t Psh = Bc * psh_stride * f16vec4;

    ಸ್ಥಿರಾಂಕ uint32_t sfshstride = (hsk <= 128) ? (Br + 8) : Br;
    ಸ್ಥಿರಾಂಕ uint32_t sfsh = Bc * sfshstride * acctype;

    ಸ್ಥಿರಾಂಕ uint32_t kvshstride = (params.shmem_staging ? std::max(hsk_pad, hsv_pad) : MatBr) / 4 + 2;
    ಸ್ಥಿರಾಂಕ uint32_t vsh_stride = MatBc / 4 * ಅಡ್ಡಸಾಲು_ವಿಭಜಿಸು;
    ಸ್ಥಿರಾಂಕ uint32_t ksh = ((kvshstride >= vsh_stride) ? (Bc * kvshstride) : (Bc * vsh_stride)) * f16vec4;

    // BF16 PVMat accumulator is f32 (no bf16 accumulator support), so pvsh is vec4 (16 bytes)
    ಸ್ಥಿರಾಂಕ uint32_t pvsh_elem_ಗಾತ್ರ = (k_type == GGML_TYPE_BF16) ? 16u : f16vec4;
    ಸ್ಥಿರಾಂಕ uint32_t osh_stride = params.ಅಡ್ಡಸಾಲು_ವಿಭಜಿಸು * MatBr / 4;
    ಸ್ಥಿರಾಂಕ uint32_t pvsh = MatBc * osh_stride * pvsh_elem_ಗಾತ್ರ;

    ಸ್ಥಿರಾಂಕ uint32_t ಇಳಿಜಾರು = Br * acctype;

    ಸ್ಥಿರಾಂಕ uint32_t ಒಟ್ಟು_ಗಾತ್ರ = tmpsh + iq_shmem + Qf + Psh + sfsh + ksh + pvsh + ಇಳಿಜಾರು;
    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ supported = ಒಟ್ಟು_ಗಾತ್ರ <= device->properties.limits.max_Compute_Shared_ಸ್ಮರಣೆ_ಗಾತ್ರ;

    VK_LOG_DEBUG("ggml_vk_flash_attn_coopmat_shmem_support(HSK=" << hsk << ", HSV=" << hsv << ", f32acc=" << f32acc << ", total_size=" << ಒಟ್ಟು_ಗಾತ್ರ << ", supported=" << supported);

    ಹಿಂತಿರುಗಿ supported;
}

ಶೂನ್ಯ ggml_vk_flash_attn(ggml_backend_vk_context * ctx, vk_context& subctx, ಸ್ಥಿರಾಂಕ ggml_tensor * q, ಸ್ಥಿರಾಂಕ ggml_tensor * k, ಸ್ಥಿರಾಂಕ ggml_tensor * v, ಸ್ಥಿರಾಂಕ ggml_tensor * ಮುಖವಾಡ, ಸ್ಥಿರಾಂಕ ggml_tensor * sinks, ggml_tensor * dst) {
    VK_LOG_DEBUG("ggml_vk_flash_attn((" << q << ", name=" << q->name << ", type=" << q->type << ", ne0=" << q->ne[0] << ", ne1=" << q->ne[1] << ", ne2=" << q->ne[2] << ", ne3=" << q->ne[3] << ", nb0=" << q->nb[0] << ", nb1=" << q->nb[1] << ", nb2=" << q->nb[2] << ", nb3=" << q->nb[3];
    std::cerr << "), (" << k << ", name=" << k->name << ", type=" << k->type << ", ne0=" << k->ne[0] << ", ne1=" << k->ne[1] << ", ne2=" << k->ne[2] << ", ne3=" << k->ne[3] << ", nb0=" << k->nb[0] << ", nb1=" << k->nb[1] << ", nb2=" << k->nb[2] << ", nb3=" << k->nb[3];
    std::cerr << "), (" << v << ", name=" << v->name << ", type=" << v->type << ", ne0=" << v->ne[0] << ", ne1=" << v->ne[1] << ", ne2=" << v->ne[2] << ", ne3=" << v->ne[3] << ", nb0=" << v->nb[0] << ", nb1=" << v->nb[1] << ", nb2=" << v->nb[2] << ", nb3=" << v->nb[3];
    std::cerr << "), (" << dst << ", name=" << dst->name << ", type=" << dst->type << ", ne0=" << dst->ne[0] << ", ne1=" << dst->ne[1] << ", ne2=" << dst->ne[2] << ", ne3=" << dst->ne[3] << ", nb0=" << dst->nb[0] << ", nb1=" << dst->nb[1] << ", nb2=" << dst->nb[2] << ", nb3=" << dst->nb[3];
    ಆದರೆ (sinks) {
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

    ಸ್ಥಿರಾಂಕ uint32_t nem0 = ಮುಖವಾಡ ? ಮುಖವಾಡ->ne[0] : 0;
    ಸ್ಥಿರಾಂಕ uint32_t nem1 = ಮುಖವಾಡ ? ಮುಖವಾಡ->ne[1] : 0;
    ಸ್ಥಿರಾಂಕ uint32_t nem2 = ಮುಖವಾಡ ? ಮುಖವಾಡ->ne[2] : 0;
    ಸ್ಥಿರಾಂಕ uint32_t nem3 = ಮುಖವಾಡ ? ಮುಖವಾಡ->ne[3] : 0;

    ಸ್ಥಿರಾಂಕ uint32_t HSK = nek0;
    ಸ್ಥಿರಾಂಕ uint32_t HSV = nev0;
    uint32_t N = neq1;
    ಸ್ಥಿರಾಂಕ uint32_t KV = nek1;

    GGML_ASSERT(ne0 == HSV);
    GGML_ASSERT(ne2 == N);

    // input tensor rows must be contiguous
    GGML_ASSERT(nbq0 == ggml_type_ಗಾತ್ರ(q->type));
    GGML_ASSERT(nbk0 == ggml_type_ಗಾತ್ರ(k->type));
    GGML_ASSERT(nbv0 == ggml_type_ಗಾತ್ರ(v->type));

    GGML_ASSERT(neq0 == HSK);

    GGML_ASSERT(neq1 == N);

    GGML_ASSERT(nev1 == nek1);

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == ಗಾತ್ರ(ದಶಮಾಂಶ));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    ಪ್ರತಿಜ್ಞಾ(dst->type == GGML_TYPE_F32);
    ಪ್ರತಿಜ್ಞಾ(q->type == GGML_TYPE_F32);
    uint32_t gqa_ಅನುಪಾತ = 1;
    uint32_t qk_ಅನುಪಾತ = neq2 / nek2;
    uint32_t workgroups_x = (uint32_t)neq1;
    uint32_t workgroups_y = (uint32_t)neq2;
    uint32_t workgroups_z = (uint32_t)neq3;

    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ f32acc = !ctx->device->fp16 || dst->op_params[3] == GGML_PREC_F32 || k->type == GGML_TYPE_BF16;

    // dequant K/V once into an f16 scratch, reordered KV layout so FA can read without a stride
    ಸ್ವಯಂಚಾಲಿತ is_dense_kv_cache = [](ಸ್ಥಿರಾಂಕ ggml_tensor * t) {
        ಹಿಂತಿರುಗಿ t->nb[0] == ggml_type_ಗಾತ್ರ(t->type) &&
               t->nb[2] == ggml_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ(t->type, t->ne[0]) &&
               t->nb[1] == t->nb[2] * t->ne[2] &&
               (t->ne[3] == 1 || t->nb[3] == t->nb[1] * t->ne[1]);
    };
    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ k_quant = k->type != GGML_TYPE_F16 && k->type != GGML_TYPE_BF16 && k->type != GGML_TYPE_F32;
    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ v_quant = v->type != GGML_TYPE_F16 && v->type != GGML_TYPE_BF16 && v->type != GGML_TYPE_F32;
    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ kv_raw = (k->type == GGML_TYPE_F16 || k->type == GGML_TYPE_BF16) && k->type == v->type;
    // gated to AMD, except RDNA4, which regressed in benchmarks
    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ kv_raw_strided = ctx->device->vendor_id == VK_VENDOR_ID_AMD &&
                                ctx->device->architecture != vk_device_architecture::AMD_RDNA4 &&
                                neq1 >= 256 &&
                                kv_raw &&
                                (k->nb[1] != (uint64_t)HSK * ggml_type_ಗಾತ್ರ(k->type) ||
                                 v->nb[1] != (uint64_t)HSV * ggml_type_ಗಾತ್ರ(v->type)) &&
                                (HSK % 8) == 0 && (HSV % 8) == 0;
    ಸ್ಥಿರಾಂಕ size_t kv_scratch_ts = kv_raw_strided ? ggml_type_ಗಾತ್ರ(k->type) : ಗಾತ್ರ(ggml_fp16_t);
    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ use_dequant_kv = ((k_quant && v_quant) || kv_raw_strided) && neq1 >= 64 &&
                                is_dense_kv_cache(k) && is_dense_kv_cache(v) &&
                                (uint64_t)ggml_nelements(k) * kv_scratch_ts <= ctx->device->properties.limits.max_Storage_ಬಫರ್_Range &&
                                (uint64_t)ggml_nelements(v) * kv_scratch_ts <= ctx->device->properties.limits.max_Storage_ಬಫರ್_Range &&
                                (kv_raw_strided ||
                                 (ctx->device->pipeline_dequant_transpose[k->type] != ಶೂನ್ಯಸೂಚಕ &&
                                  ctx->device->pipeline_dequant_transpose[v->type] != ಶೂನ್ಯಸೂಚಕ)) &&
                                // coopmat2 path does not benefit from the f16 scratch
                                !ctx->device->coopmat2 &&
                                // Intel Xe1 regresses, see PR 25494
                                (ctx->device->vendor_id != VK_VENDOR_ID_INTEL ||
                                 (ctx->device->coopmat_support && ctx->device->architecture != vk_device_architecture::INTEL_XE1));
    ಸ್ಥಿರಾಂಕ ggml_type k_type_eff = (use_dequant_kv && !kv_raw_strided) ? GGML_TYPE_F16 : k->type;
    ಸ್ಥಿರಾಂಕ ggml_type v_type_eff = (use_dequant_kv && !kv_raw_strided) ? GGML_TYPE_F16 : v->type;

    // For scalar/coopmat1 FA, we can use the "large" size to accommodate qga.
    // For coopmat2 FA, we always use the small size (which is still pretty large for gqa).
    vk_fa_tuning_params tuning_params = get_fa_tuning_params(ctx->device, HSK, HSV, 512, KV, k_type_eff, v_type_eff, f32acc);
    ಸ್ಥಿರಾಂಕ uint32_t max_gqa = std::min(tuning_params.ಖಂಡ_ಅಡ್ಡಸಾಲುಗಳು, 32u);

    ಆದರೆ (N <= 8 && qk_ಅನುಪಾತ > 1 && qk_ಅನುಪಾತ <= max_gqa &&
        qk_ಅನುಪಾತ * nek2 == neq2 && nek2 == nev2 && nem2 <= 1) {
        // grouped query attention - make the N dimension equal to gqa_ratio, reduce
        // workgroups proportionally in y dimension. The shader will detect gqa_ratio > 1
        // and change addressing calculations to index Q's dimension 2.
        gqa_ಅನುಪಾತ = qk_ಅನುಪಾತ;
        N = gqa_ಅನುಪಾತ;
        workgroups_y /= gqa_ಅನುಪಾತ;
    }

    tuning_params = get_fa_tuning_params(ctx->device, HSK, HSV, N, KV, k_type_eff, v_type_eff, f32acc);

    ದಶಮಾಂಶ scale         = 1.0f;
    ದಶಮಾಂಶ max_bias      = 0.0f;
    ದಶಮಾಂಶ logit_softcap = 0.0f;

    ಸ್ಮರಣೆ_ನಕಲು(&scale,         (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) dst->op_params + 0, ಗಾತ್ರ(ದಶಮಾಂಶ));
    ಸ್ಮರಣೆ_ನಕಲು(&max_bias,      (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) dst->op_params + 1, ಗಾತ್ರ(ದಶಮಾಂಶ));
    ಸ್ಮರಣೆ_ನಕಲು(&logit_softcap, (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) dst->op_params + 2, ಗಾತ್ರ(ದಶಮಾಂಶ));

    ಆದರೆ (logit_softcap != 0) {
        scale /= logit_softcap;
    }

    // Sparse mask hint (op_params[4]): compact the <= n_kv_max finite positions and gather only those.
    ಸ್ಥಿರಾಂಕ int32_t n_kv_max = ಮುಖವಾಡ ? ggml_get_op_params_i32(dst, 4) : 0;
    ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ disable_sparse = getenv("GGML_VK_FA_SPARSE_DISABLE") != ಶೂನ್ಯಸೂಚಕ;
    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ kv_f16 = k_type_eff == GGML_TYPE_F16 && v_type_eff == GGML_TYPE_F16;
    // cm2 dense is fast, so it needs a larger reduction to win.
    // With quantized K/V, sparse only breaks even around 16x (measured on RDNA3/RDNA4).
    ಸ್ಥಿರಾಂಕ int64_t min_ಅನುಪಾತ = tuning_params.path == FA_COOPMAT2 ? 4 : (kv_f16 ? 2 : 16);
    // coopmat2 vector decode requires 8B strides.
    ಸ್ವಯಂಚಾಲಿತ sparse_gather_aligned = [](ಸ್ಥಿರಾಂಕ ggml_tensor * t) {
        ಹಿಂತಿರುಗಿ (t->type != GGML_TYPE_F16 && t->type != GGML_TYPE_BF16) ||
               (t->nb[1] | t->nb[2] | t->nb[3]) % (4 * ಗಾತ್ರ(ggml_fp16_t)) == 0;
    };
    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ use_sparse = !disable_sparse && n_kv_max > 0 && ಮುಖವಾಡ &&
                            max_bias == 0.0f && logit_softcap == 0.0f &&
                            (tuning_params.path != FA_COOPMAT2 || (sparse_gather_aligned(k) && sparse_gather_aligned(v))) &&
                            nem0 == KV &&
                            (int64_t)KV >= std::max<int64_t>(4096, min_ಅನುಪಾತ * (int64_t)n_kv_max) &&
                            (gqa_ಅನುಪಾತ > 1 || (tuning_params.path == FA_SCALAR && N == 1));

    ಸ್ಥಿರಾಂಕ uint32_t q_stride = (uint32_t)(nbq1 / ggml_type_ಗಾತ್ರ(q->type));
    uint32_t k_stride = (uint32_t)(nbk1 / ggml_type_ಗಾತ್ರ(k->type));
    uint32_t v_stride = (uint32_t)(nbv1 / ggml_type_ಗಾತ್ರ(v->type));

    // For F32, the shader treats it as a block of size 4 (for vec4 loads)
    ಆದರೆ (k->type == GGML_TYPE_F32) {
        k_stride /= 4;
    }
    ಆದರೆ (v->type == GGML_TYPE_F32) {
        v_stride /= 4;
    }

    uint32_t nbk2_eff = (uint32_t)nbk2, nbk3_eff = (uint32_t)nbk3;
    uint32_t nbv2_eff = (uint32_t)nbv2, nbv3_eff = (uint32_t)nbv3;
    ಆದರೆ (use_dequant_kv) {
        k_stride = HSK;
        v_stride = HSV;
        nbk2_eff = (uint32_t)((uint64_t)HSK * KV * ಗಾತ್ರ(ggml_fp16_t));
        nbk3_eff = (uint32_t)((uint64_t)HSK * KV * nek2 * ಗಾತ್ರ(ggml_fp16_t));
        nbv2_eff = (uint32_t)((uint64_t)HSV * KV * ಗಾತ್ರ(ggml_fp16_t));
        nbv3_eff = (uint32_t)((uint64_t)HSV * KV * nev2 * ಗಾತ್ರ(ggml_fp16_t));
    }
    ಸ್ಥಿರಾಂಕ uint32_t alignment = tuning_params.ಖಂಡ_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ;
    ತಾರ್ಕಿಕ aligned = (KV % alignment) == 0 &&
                   // the "aligned" shader variant will forcibly align strides, for performance
                   (q_stride & 7) == 0 && (k_stride & 7) == 0 && (v_stride & 7) == 0;

    // Need to use the coopmat2 variant that clamps loads when HSK/HSV aren't sufficiently aligned.
    ಆದರೆ (((HSK | HSV) % 16) != 0 && tuning_params.path == FA_COOPMAT2) {
        aligned = ಸುಳ್ಳು;
    }

    // Only use mask opt when the mask is fairly large. This hasn't been tuned extensively.
    ತಾರ್ಕಿಕ use_ಮುಖವಾಡ_opt = ಮುಖವಾಡ && !use_sparse && nem1 >= 32 && nem0 * nem1 > 32768 && nem0 >= tuning_params.ಖಂಡ_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ * 16
                        && (ctx->device->architecture != vk_device_architecture::AMD_GCN || HSK > 256 || HSV > 256);
    vk_fa_pipeline_state fa_pipeline_state = get_fa_pipeline_state(ctx->device, tuning_params, HSK, HSV, aligned, f32acc,
                                                                   ಮುಖವಾಡ != ಶೂನ್ಯಸೂಚಕ, use_ಮುಖವಾಡ_opt, logit_softcap != 0, use_sparse, k_type_eff, v_type_eff);

    vk_pipeline pipeline = ಶೂನ್ಯಸೂಚಕ;

    ತಾರ್ಕಿಕ xe_fa_opt = ಸುಳ್ಳು;
    ತಾರ್ಕಿಕ fa_copy_qstate = ಸುಳ್ಳು;
    ತಾರ್ಕಿಕ xe_fa_supported_platform =
        (ctx->device.get()->architecture == INTEL_XE2 && ctx->device.get()->properties.deviceID != 0xFD80 && ctx->device.get()->properties.deviceID != 0xFD81) ||
        (ctx->device.get()->architecture == INTEL_XE1 && ctx->device.get()->coopmat_support && ctx->device.get()->uma);
    ತಾರ್ಕಿಕ xe_fa_supported_usage = neq0 % 32 == 0 && nev0 % 16 == 0 && q->nb[1] > q->nb[2] && k->nb[1] > k->nb[2] && v->nb[1] > v->nb[2] && ಮುಖವಾಡ != ಶೂನ್ಯಸೂಚಕ;
    ತಾರ್ಕಿಕ xe_fa_supported_dtype = q->type == GGML_TYPE_F32 && k->type == GGML_TYPE_F16 && v->type == GGML_TYPE_F16 && (ಮುಖವಾಡ != ಶೂನ್ಯಸೂಚಕ && ಮುಖವಾಡ->type == GGML_TYPE_F16);
    std::pair<vk_pipeline, vk_pipeline> xe_fa_pipeline_dual_phases = { ಶೂನ್ಯಸೂಚಕ , ಶೂನ್ಯಸೂಚಕ };
    vk_pipeline xe_fa_pipeline = ಶೂನ್ಯಸೂಚಕ;
    size_t ಗಾತ್ರ_p = 0;
    size_t ಗಾತ್ರ_ಗುಂಪು_max = 0;

    {
        std::lock_guard<std::mutex> guard(ctx->device->compile_mutex);
        ಸ್ವಯಂಚಾಲಿತ &pipelines = ctx->device->pipeline_flash_attn_f32_f16;
        ಸ್ವಯಂಚಾಲಿತ it = pipelines.find(fa_pipeline_state);
        ಆದರೆ (it != pipelines.end()) {
            pipeline = it->second;
        } ಇಲ್ಲದಿದ್ದರೆ {
            pipelines[fa_pipeline_state] = pipeline = std::make_shared<vk_pipeline_struct>();
        }
    }

    ಪ್ರತಿಜ್ಞಾ(pipeline);
    // Compile early to initialize wg_denoms.
    ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);

    uint32_t ವಿಭಜಿಸು_kv = KV;
    uint32_t ವಿಭಜಿಸು_k = 1;

    // Intel Alchemist prefers more workgroups
    ಸ್ಥಿರಾಂಕ uint32_t shader_core_ಎಣಿಕೆ_multiplier = (ctx->device->vendor_id == VK_VENDOR_ID_INTEL && ctx->device->architecture != INTEL_XE2) ? 2 : 1;

    // Use a placeholder core count if one isn't available. split_k is a big help for perf.
    ಸ್ಥಿರಾಂಕ uint32_t shader_core_ಎಣಿಕೆ = ctx->device->shader_core_ಎಣಿಕೆ ? ctx->device->shader_core_ಎಣಿಕೆ * shader_core_ಎಣಿಕೆ_multiplier : 16;

    ಸ್ಥಿರಾಂಕ uint32_t Br = fa_pipeline_state.Br;
    ಸ್ಥಿರಾಂಕ uint32_t Bc = fa_pipeline_state.Bc;

    GGML_ASSERT(Br == pipeline->wg_denoms[0]);
    ಸ್ಥಿರಾಂಕ uint32_t Tr = CEIL_DIV(N, Br);

    // Try to use split_k when KV is large enough to be worth the overhead.
    // Sparse: split_kv carries n_kv_max, split_k partitions its blocks for occupancy.
    ಆದರೆ (use_sparse) {
        ವಿಭಜಿಸು_kv = (uint32_t)n_kv_max;
        ಸ್ಥಿರಾಂಕ uint32_t ಒಟ್ಟು_ಖಂಡಗಳು = CEIL_DIV((uint32_t)n_kv_max, Bc);
        ಸ್ಥಿರಾಂಕ uint32_t ತಳ_wgs = (gqa_ಅನುಪಾತ > 1 ? workgroups_x : Tr) * workgroups_y * workgroups_z;
        ಆದರೆ (ತಳ_wgs < shader_core_ಎಣಿಕೆ * 2) {
            ವಿಭಜಿಸು_k = shader_core_ಎಣಿಕೆ * 2 / ತಳ_wgs;
        }
        ವಿಭಜಿಸು_k = std::max(1u, std::min(ವಿಭಜಿಸು_k, ಒಟ್ಟು_ಖಂಡಗಳು));
        // Match the shader's per-split block count so no split is empty.
        ಸ್ಥಿರಾಂಕ uint32_t ಪ್ರತಿ_ಖಂಡಗಳು = CEIL_DIV(ಒಟ್ಟು_ಖಂಡಗಳು, ವಿಭಜಿಸು_k);
        ವಿಭಜಿಸು_k = CEIL_DIV(ಒಟ್ಟು_ಖಂಡಗಳು, ಪ್ರತಿ_ಖಂಡಗಳು);
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (gqa_ಅನುಪಾತ > 1 && workgroups_x <= Br) {
        ವಿಭಜಿಸು_k = shader_core_ಎಣಿಕೆ * 2 / (workgroups_x * workgroups_y * workgroups_z);
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (gqa_ಅನುಪಾತ <= 1) {
        uint32_t ಒಟ್ಟು_wgs_no_ವಿಭಜಿಸು = Tr * workgroups_y * workgroups_z;
        ಆದರೆ (ಒಟ್ಟು_wgs_no_ವಿಭಜಿಸು < shader_core_ಎಣಿಕೆ * 2) {
            ವಿಭಜಿಸು_k = shader_core_ಎಣಿಕೆ * 2 / ಒಟ್ಟು_wgs_no_ವಿಭಜಿಸು;
        }
    }

    ಆದರೆ (!use_sparse && ವಿಭಜಿಸು_k > 1) {
        // Try to evenly split KV into split_k chunks, but it needs to be a multiple
        // of "align", so recompute split_k based on that.
        ವಿಭಜಿಸು_kv = ROUNDUP_POW2(std::max(1u, KV / ವಿಭಜಿಸು_k), alignment);
        ವಿಭಜಿಸು_k = CEIL_DIV(KV, ವಿಭಜಿಸು_kv);
        xe_fa_opt = xe_fa_supported_platform && xe_fa_supported_usage && xe_fa_supported_dtype;
        ಆದರೆ (xe_fa_opt) {
            ಸ್ಥಿರಾಂಕ uint32_t ವಿಭಜಿಸು_p_ಗಾತ್ರ = 32;
            ಸ್ಥಿರಾಂಕ size_t max_dim = (nek1 + ವಿಭಜಿಸು_p_ಗಾತ್ರ - 1) / ವಿಭಜಿಸು_p_ಗಾತ್ರ;
            ಸ್ಥಿರಾಂಕ size_t p_dim = max_dim * ವಿಭಜಿಸು_p_ಗಾತ್ರ;
#ಆದರೆ defined(VK_KHR_cooperative_ಮಾತೃಕೆ) && defined(GGML_VULKAN_COOPMAT_GLSLC_SUPPORT)
            std::lock_guard<std::mutex> guard(ctx->device->compile_mutex);
            ಸ್ವಯಂಚಾಲಿತ& pipelines = ctx->device->pipeline_xe_fa_decode_dual_phases;
            ಸ್ವಯಂಚಾಲಿತ it = pipelines.find({ (uint32_t)neq0, (uint32_t)nev0, qk_ಅನುಪಾತ, (uint32_t)neq1 });
            ಆದರೆ (it != pipelines.end()) {
                xe_fa_pipeline_dual_phases = it->second;
            } ಇಲ್ಲದಿದ್ದರೆ {
                pipelines[{(uint32_t)neq0, (uint32_t)nev0, qk_ಅನುಪಾತ, (uint32_t)neq1}] = xe_fa_pipeline_dual_phases = std::make_pair(std::make_shared<vk_pipeline_struct>(), std::make_shared<vk_pipeline_struct>());
            }
#ಆದರೆ_ಅಂತ್ಯ
            ಆದರೆ (xe_fa_pipeline_dual_phases.ಮೊದಲ == ಶೂನ್ಯಸೂಚಕ || xe_fa_pipeline_dual_phases.second == ಶೂನ್ಯಸೂಚಕ) {
                xe_fa_opt = ಸುಳ್ಳು;
                fa_copy_qstate = ಸುಳ್ಳು;
            } ಇಲ್ಲದಿದ್ದರೆ {
                ಗಾತ್ರ_p = neq1 * neq2 * p_dim * neq3 * ಗಾತ್ರ(ggml_fp16_t);
                ಗಾತ್ರ_ಗುಂಪು_max = neq1 * neq2 * max_dim * neq3 * ಗಾತ್ರ(ದಶಮಾಂಶ);
                size_t ತಾತ್ಕಾಲಿಕ_ಗಾತ್ರ = ggml_nelements(q) * ಗಾತ್ರ(ggml_fp16_t) + ಗಾತ್ರ_p + ಗಾತ್ರ_ಗುಂಪು_max;
                fa_copy_qstate = ನಿಜ;
                ಆದರೆ (ctx->prealloc_ಗಾತ್ರ_x < ತಾತ್ಕಾಲಿಕ_ಗಾತ್ರ) {
                    ctx->prealloc_ಗಾತ್ರ_x = ತಾತ್ಕಾಲಿಕ_ಗಾತ್ರ;
                    ggml_vk_preallocate_buffers(ctx, subctx);
                }

                ಆದರೆ (ctx->prealloc_x_need_sync) {
                    ggml_vk_sync_buffers(ctx, subctx);
                }
            }
        }
    }

    ಆದರೆ (xe_fa_opt == ನಿಜ) {
        use_ಮುಖವಾಡ_opt = ಸುಳ್ಳು;
    }

    // Reserve space for split_k temporaries. For each split x batch, we need to store the O matrix (D x ne1)
    // and the per-row m and L values (ne1 rows). We store all the matrices first, followed by the rows.
    // For matrices, the order is (inner to outer) [HSV, ne1, k, ne2, ne3].
    // For L/M, the order is (inner to outer) [ne1, k, ne2, ne3].
    ಸ್ಥಿರಾಂಕ uint64_t ವಿಭಜಿಸು_k_ಗಾತ್ರ = ವಿಭಜಿಸು_k > 1 ? (HSV * ne1 * ಗಾತ್ರ(ದಶಮಾಂಶ) + ne1 * ಗಾತ್ರ(ದಶಮಾಂಶ) * 2) * ವಿಭಜಿಸು_k * ne2 * ne3 : 0;
    ಆದರೆ (ವಿಭಜಿಸು_k_ಗಾತ್ರ > ctx->device->properties.limits.max_Storage_ಬಫರ್_Range) {
        GGML_ABORT("Requested preallocation size is too large");
    }
    ಆದರೆ (ctx->prealloc_ಗಾತ್ರ_ವಿಭಜಿಸು_k < ವಿಭಜಿಸು_k_ಗಾತ್ರ) {
        ctx->prealloc_ಗಾತ್ರ_ವಿಭಜಿಸು_k = ವಿಭಜಿಸು_k_ಗಾತ್ರ;
        ggml_vk_preallocate_buffers(ctx, subctx);
    }

    ಸ್ಥಿರಾಂಕ uint32_t ಮುಖವಾಡ_opt_ಅಂಕೆ_dwords = CEIL_DIV(nem0, 16 * Bc);
    ಸ್ಥಿರಾಂಕ uint64_t ಮುಖವಾಡ_opt_ಗಾತ್ರ = ಗಾತ್ರ(uint32_t) * ಮುಖವಾಡ_opt_ಅಂಕೆ_dwords * CEIL_DIV(nem1, Br) * nem2 * nem3;

    vk_pipeline pipeline_fa_ಮುಖವಾಡ_opt = ಶೂನ್ಯಸೂಚಕ;
    ಆದರೆ (use_ಮುಖವಾಡ_opt) {
        {
            std::lock_guard<std::mutex> guard(ctx->device->compile_mutex);
            ಸ್ವಯಂಚಾಲಿತ &pipelines = ctx->device->pipeline_fa_ಮುಖವಾಡ_opt;
            ಸ್ವಯಂಚಾಲಿತ it = pipelines.find({Br, Bc});
            ಆದರೆ (it != pipelines.end()) {
                pipeline_fa_ಮುಖವಾಡ_opt = it->second;
            } ಇಲ್ಲದಿದ್ದರೆ {
                pipelines[{Br, Bc}] = pipeline_fa_ಮುಖವಾಡ_opt = std::make_shared<vk_pipeline_struct>();
            }
        }
        ಪ್ರತಿಜ್ಞಾ(pipeline_fa_ಮುಖವಾಡ_opt);
        ggml_pipeline_request_descriptor_sets(ctx, pipeline_fa_ಮುಖವಾಡ_opt, 1);

        ಆದರೆ (ctx->prealloc_ಗಾತ್ರ_y < ಮುಖವಾಡ_opt_ಗಾತ್ರ) {
            ctx->prealloc_ಗಾತ್ರ_y = ಮುಖವಾಡ_opt_ಗಾತ್ರ;
            ggml_vk_preallocate_buffers(ctx, subctx);
        }
        ಆದರೆ (ctx->prealloc_y_need_sync) {
            ggml_vk_sync_buffers(ctx, subctx);
        }
    }

    // Sparse index scratch reuses prealloc_y (mutually exclusive with mask opt).
    ಸ್ಥಿರಾಂಕ uint64_t sparse_idx_ಗಾತ್ರ = use_sparse
        ? ಗಾತ್ರ(int32_t) * (uint64_t)n_kv_max * nem1 * nem2 * nem3
        : 0;
    vk_pipeline sparse_compact_pipeline = ctx->device->fa_sparse_compact_use_subgroups
        ? ctx->device->pipeline_fa_sparse_compact_subgroup
        : ctx->device->pipeline_fa_sparse_compact;
    ಆದರೆ (use_sparse) {
        ggml_pipeline_request_descriptor_sets(ctx, sparse_compact_pipeline, 1);
        ಆದರೆ (ctx->prealloc_ಗಾತ್ರ_y < sparse_idx_ಗಾತ್ರ) {
            ctx->prealloc_ಗಾತ್ರ_y = sparse_idx_ಗಾತ್ರ;
            ggml_vk_preallocate_buffers(ctx, subctx);
        }
        ಆದರೆ (ctx->prealloc_y_need_sync) {
            ggml_vk_sync_buffers(ctx, subctx);
        }
    }

    ಸ್ಥಿರಾಂಕ uint32_t n_ತಲೆ_kv   = neq2;
    ಸ್ಥಿರಾಂಕ uint32_t n_ತಲೆ_log2 = 1u << (uint32_t) floorf(log2f((ದಶಮಾಂಶ) n_ತಲೆ_kv));
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ m0 = powf(2.0f, -(max_bias       ) / n_ತಲೆ_log2);
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ m1 = powf(2.0f, -(max_bias / 2.0f) / n_ತಲೆ_log2);

    vk_subbuffer q_buf = ggml_vk_tensor_subbuffer(ctx, q);
    vk_subbuffer k_buf = ggml_vk_tensor_subbuffer(ctx, k);
    vk_subbuffer v_buf = ggml_vk_tensor_subbuffer(ctx, v);
    vk_subbuffer dst_buf = ggml_vk_tensor_subbuffer(ctx, dst);
    vk_subbuffer ಮುಖವಾಡ_buf = ಮುಖವಾಡ ? ggml_vk_tensor_subbuffer(ctx, ಮುಖವಾಡ) : q_buf;
    vk_subbuffer sinks_buf = sinks ? ggml_vk_tensor_subbuffer(ctx, sinks) : q_buf;
    vk_subbuffer ಮುಖವಾಡ_opt_buf = use_ಮುಖವಾಡ_opt ? ggml_vk_subbuffer(ctx, ctx->prealloc_y, 0) : q_buf;
    vk_subbuffer sparse_buf = use_sparse ? ggml_vk_subbuffer(ctx, ctx->prealloc_y, 0) : q_buf;
    ಆದರೆ (use_ಮುಖವಾಡ_opt || use_sparse) {
        // the mask opt bits and the sparse index list overwrite a matmul input converted into prealloc_y
        ctx->prealloc_y_ಕೊನೆಯ_pipeline_ಬಳಸಿದ = ಶೂನ್ಯಸೂಚಕ;
        ctx->prealloc_y_ಕೊನೆಯ_tensor_ಬಳಸಿದ = ಶೂನ್ಯಸೂಚಕ;
    }

    ಆದರೆ (use_dequant_kv) {
        ಸ್ಥಿರಾಂಕ uint64_t k_sz = (uint64_t)ggml_nelements(k) * kv_scratch_ts;
        ಸ್ಥಿರಾಂಕ uint64_t v_sz = (uint64_t)ggml_nelements(v) * kv_scratch_ts;
        ಆದರೆ (ctx->prealloc_ಗಾತ್ರ_x < k_sz + v_sz) {
            ctx->prealloc_ಗಾತ್ರ_x = k_sz + v_sz;
            ggml_vk_preallocate_buffers(ctx, subctx);
        }
        vk_subbuffer k_dst = vk_subbuffer{ ctx->prealloc_x, 0,    k_sz };
        vk_subbuffer v_dst = vk_subbuffer{ ctx->prealloc_x, k_sz, v_sz };
        ಸ್ಥಿರಾಂಕ uint32_t k_nel = (uint32_t)ggml_nelements(k);
        ಸ್ಥಿರಾಂಕ uint32_t v_nel = (uint32_t)ggml_nelements(v);
        ಆದರೆ (kv_raw_strided) {
            vk_pipeline cp_k = ggml_vk_get_cpy_pipeline(ctx, k, ಶೂನ್ಯಸೂಚಕ, k->type);
            vk_pipeline cp_v = ggml_vk_get_cpy_pipeline(ctx, v, ಶೂನ್ಯಸೂಚಕ, v->type);
            ggml_pipeline_request_descriptor_sets(ctx, cp_k, 1);
            ggml_pipeline_request_descriptor_sets(ctx, cp_v, 1);
            ಆದರೆ (ctx->prealloc_x_need_sync) {
                ggml_vk_sync_buffers(ctx, subctx);
            }
            ಸ್ವಯಂಚಾಲಿತ make_pc = [](uint32_t hs, uint32_t nh, uint32_t kv, uint32_t nel) {
                vk_op_unary_ತಳ್ಳು_constants pc{};
                pc.ne = nel;
                pc.ne00 = hs; pc.ne01 = nh;      pc.ne02 = kv; pc.ne03 = nel / (hs * nh * kv);
                pc.nb00 = 1;  pc.nb01 = hs;      pc.nb02 = hs * nh; pc.nb03 = hs * nh * kv;
                pc.ne10 = hs; pc.ne11 = nh;      pc.ne12 = kv; pc.ne13 = pc.ne03;
                pc.nb10 = 1;  pc.nb11 = hs * kv; pc.nb12 = hs; pc.nb13 = hs * kv * nh;
                init_pushconst_fastdiv(pc);
                ಹಿಂತಿರುಗಿ pc;
            };
            ಸ್ವಯಂಚಾಲಿತ cpy_elems = [](uint32_t ne) -> std::array<uint32_t, 3> {
                ಆದರೆ (ne > 262144) { ಹಿಂತಿರುಗಿ { 512, 512, CEIL_DIV(ne, 262144) }; }
                ಆದರೆ (ne > 512)    { ಹಿಂತಿರುಗಿ { 512, CEIL_DIV(ne, 512), 1 }; }
                ಹಿಂತಿರುಗಿ { ne, 1, 1 };
            };
            ggml_vk_dispatch_pipeline(ctx, subctx, cp_k, { k_buf, k_dst },
                make_pc((uint32_t)HSK, (uint32_t)nek2, (uint32_t)KV, k_nel), cpy_elems(k_nel));
            ggml_vk_dispatch_pipeline(ctx, subctx, cp_v, { v_buf, v_dst },
                make_pc((uint32_t)HSV, (uint32_t)nev2, (uint32_t)KV, v_nel), cpy_elems(v_nel));
        } ಇಲ್ಲದಿದ್ದರೆ {
            vk_pipeline tr_k = ctx->device->pipeline_dequant_transpose[k->type];
            vk_pipeline tr_v = ctx->device->pipeline_dequant_transpose[v->type];
            ggml_pipeline_request_descriptor_sets(ctx, tr_k, 1);
            ggml_pipeline_request_descriptor_sets(ctx, tr_v, 1);
            ಆದರೆ (ctx->prealloc_x_need_sync) {
                ggml_vk_sync_buffers(ctx, subctx);
            }
            { ಸ್ಥಿರಾಂಕ std::vector<uint32_t> pc = { (uint32_t)HSK, (uint32_t)nek2, (uint32_t)KV, 0, k_nel };
              ggml_vk_dispatch_pipeline(ctx, subctx, tr_k, { k_buf, k_dst }, pc, { k_nel, 1, 1 }); }
            { ಸ್ಥಿರಾಂಕ std::vector<uint32_t> pc = { (uint32_t)HSV, (uint32_t)nev2, (uint32_t)KV, 0, v_nel };
              ggml_vk_dispatch_pipeline(ctx, subctx, tr_v, { v_buf, v_dst }, pc, { v_nel, 1, 1 }); }
        }
        ggml_vk_sync_buffers(ctx, subctx);
        k_buf = k_dst;
        v_buf = v_dst;
    }

    uint32_t ಮುಖವಾಡ_n_ತಲೆ_log2 = ((sinks != ಶೂನ್ಯಸೂಚಕ) << 24) | n_ತಲೆ_log2;

    ಆದರೆ (use_ಮುಖವಾಡ_opt)
    {
        ಸ್ಥಿರಾಂಕ vk_op_flash_attn_ಮುಖವಾಡ_opt_ತಳ್ಳು_constants opt_pc = {
            nem0,
            nem1,
            nem2,
            (uint32_t)(ಮುಖವಾಡ->nb[1] / ಗಾತ್ರ(ggml_fp16_t)),
            (uint32_t)(ಮುಖವಾಡ->nb[2] / ಗಾತ್ರ(ggml_fp16_t)),
            (uint32_t)(ಮುಖವಾಡ->nb[3] / ಗಾತ್ರ(ggml_fp16_t)),
            ಮುಖವಾಡ_opt_ಅಂಕೆ_dwords,
            ಮುಖವಾಡ_opt_ಅಂಕೆ_dwords * CEIL_DIV(nem1, Br),
            ಮುಖವಾಡ_opt_ಅಂಕೆ_dwords * CEIL_DIV(nem1, Br) * nem2,
        };

        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline_fa_ಮುಖವಾಡ_opt,
                                  { ಮುಖವಾಡ_buf, ಮುಖವಾಡ_opt_buf }, opt_pc,
                                  { ಮುಖವಾಡ_opt_ಅಂಕೆ_dwords, CEIL_DIV(nem1, Br), nem2 * nem3 });
        ggml_vk_sync_buffers(ctx, subctx);
    }

    ಆದರೆ (use_sparse)
    {
        ಸ್ಥಿರಾಂಕ vk_op_flash_attn_sparse_compact_ತಳ್ಳು_constants sc_pc = {
            KV,
            nem1,
            nem2,
            (uint32_t)(ಮುಖವಾಡ->nb[1] / ಗಾತ್ರ(ggml_fp16_t)),
            (uint32_t)(ಮುಖವಾಡ->nb[2] / ಗಾತ್ರ(ggml_fp16_t)),
            (uint32_t)(ಮುಖವಾಡ->nb[3] / ಗಾತ್ರ(ggml_fp16_t)),
            (uint32_t)n_kv_max,
        };

        ggml_vk_dispatch_pipeline(ctx, subctx, sparse_compact_pipeline,
                                  { ಮುಖವಾಡ_buf, sparse_buf }, sc_pc,
                                  { nem1, nem2, nem3 });
        ggml_vk_sync_buffers(ctx, subctx);
    }

    ಸ್ಥಿರಾಂಕ vk_flash_attn_ತಳ್ಳು_constants pc = { N, KV,
                                              (uint32_t)ne1, (uint32_t)ne2, (uint32_t)ne3,
                                              (uint32_t)neq2, (uint32_t)neq3,
                                              (uint32_t)nek2, (uint32_t)nek3,
                                              (uint32_t)nev2, (uint32_t)nev3,
                                              nem1, nem2, nem3,
                                              q_stride, (uint32_t)nbq2, (uint32_t)nbq3,
                                              k_stride, nbk2_eff, nbk3_eff,
                                              v_stride, nbv2_eff, nbv3_eff,
                                              scale, max_bias, logit_softcap,
                                              ಮುಖವಾಡ_n_ತಲೆ_log2, m0, m1,
                                              gqa_ಅನುಪಾತ, ವಿಭಜಿಸು_kv, ವಿಭಜಿಸು_k };

    ಆದರೆ (xe_fa_opt && ವಿಭಜಿಸು_k > 1) {
        ಸ್ವಯಂಚಾಲಿತ ಊರ್ಧ್ವ_power_ದ_2 = [&](uint32_t ಒಳ) {
            GGML_ASSERT(ಒಳ != 0);
            ಆದರೆ (ಒಳ <= 1) ಹಿಂತಿರುಗಿ 1u;
            uint32_t ret = ಒಳ - 1;
            ret |= ret >> 1;
            ret |= ret >> 2;
            ret |= ret >> 4;
            ret |= ret >> 8;
            ret |= ret >> 16;
            ಹಿಂತಿರುಗಿ ret + 1;
        };
        ಸ್ವಯಂಚಾಲಿತ ಗಮ್ಯ_fp16_vk_0 = ggml_vk_get_ಗಮ್ಯ_fp16(ctx, q->type);
        ಸ್ಥಿರಾಂಕ uint32_t ಹೊರಹೋಗು_dim_ಪ್ರತಿ_wg = qk_ಅನುಪಾತ > 16 ? 8 : 16;
        size_t x_ne = ggml_nelements(q);
        size_t ತಾತ್ಕಾಲಿಕ_buf_ಸ್ಥಾನಾಂತರ = 0;
        uint32_t ತಲೆ_stride_k = uint32_t(nbk2 / ggml_type_ಗಾತ್ರ(k->type));
        uint32_t ತಲೆ_stride_v = uint32_t(nbv2 / ggml_type_ಗಾತ್ರ(v->type));
        uint32_t batch_stride_q = uint32_t(nbq3 / ggml_type_ಗಾತ್ರ(q->type));
        uint32_t batch_stride_k = uint32_t(nbk3 / ggml_type_ಗಾತ್ರ(k->type));
        uint32_t batch_stride_v = uint32_t(nbv3 / ggml_type_ಗಾತ್ರ(v->type));
        uint32_t batch_stride_m = ಮುಖವಾಡ ? uint32_t(ಮುಖವಾಡ->nb[3] / ggml_type_ಗಾತ್ರ(ಮುಖವಾಡ->type)) : 0u;
        uint32_t batch_stride_o = uint32_t(nb3 / ggml_type_ಗಾತ್ರ(dst->type));
        vk_fa_xe_opt_ತಳ್ಳು_constants pc_ph1 = { (uint32_t)nek1, (uint32_t)neq1, (uint32_t)neq2, (uint32_t)nek2, qk_ಅನುಪಾತ, 1, (sinks != ಶೂನ್ಯಸೂಚಕ) ? 1u : 0u, (uint32_t)k_stride, ತಲೆ_stride_k,
            batch_stride_q, batch_stride_k, batch_stride_v, batch_stride_m, batch_stride_o, scale };
        vk_fa_xe_opt_ತಳ್ಳು_constants pc_ph2 = pc_ph1;
        pc_ph2.nbkv_tok = v_stride;
        pc_ph2.nbkv_ತಲೆ = ತಲೆ_stride_v;
        vk_subbuffer q_ತಾತ್ಕಾಲಿಕ_buf = fa_copy_qstate ? ggml_vk_subbuffer(ctx, ctx->prealloc_x, ತಾತ್ಕಾಲಿಕ_buf_ಸ್ಥಾನಾಂತರ) : q_buf;
        ತಾತ್ಕಾಲಿಕ_buf_ಸ್ಥಾನಾಂತರ += fa_copy_qstate ? x_ne * ಗಾತ್ರ(ggml_fp16_t) : 0;
        vk_subbuffer p_ತಾತ್ಕಾಲಿಕ_buf = ggml_vk_subbuffer(ctx, ctx->prealloc_x, ತಾತ್ಕಾಲಿಕ_buf_ಸ್ಥಾನಾಂತರ);
        ತಾತ್ಕಾಲಿಕ_buf_ಸ್ಥಾನಾಂತರ += ಗಾತ್ರ_p;
        vk_subbuffer max_ತಾತ್ಕಾಲಿಕ_buf = ggml_vk_subbuffer(ctx, ctx->prealloc_x, ತಾತ್ಕಾಲಿಕ_buf_ಸ್ಥಾನಾಂತರ);
        ತಾತ್ಕಾಲಿಕ_buf_ಸ್ಥಾನಾಂತರ += ಗಾತ್ರ_ಗುಂಪು_max;
        uint32_t xe_native_sub_ಗುಂಪು_ಗಾತ್ರ = ctx->device.get()->architecture == INTEL_XE1 ? 8 : 16;
        uint32_t aligned_gqa_ಅನುಪಾತ = ಊರ್ಧ್ವ_power_ದ_2(qk_ಅನುಪಾತ);
        uint32_t ಹೊರಹೋಗು_ಪ್ರತಿ_wg_ph1 = std::min(256u / xe_native_sub_ಗುಂಪು_ಗಾತ್ರ, (uint32_t)neq1);
        uint32_t ಹೊರಹೋಗು_ಪ್ರತಿ_wg_ph2 = std::min(std::max(16u / aligned_gqa_ಅನುಪಾತ, 1u), (uint32_t)neq1);
        uint32_t ph1_wg = ((neq1 + ಹೊರಹೋಗು_ಪ್ರತಿ_wg_ph1 - 1) / ಹೊರಹೋಗು_ಪ್ರತಿ_wg_ph1) * nek2;
        uint32_t ph2_wg = ((neq1 + ಹೊರಹೋಗು_ಪ್ರತಿ_wg_ph2 - 1) / ಹೊರಹೋಗು_ಪ್ರತಿ_wg_ph2) * ne0 / ಹೊರಹೋಗು_dim_ಪ್ರತಿ_wg;
        ಆದರೆ (fa_copy_qstate) {
            ಸ್ಥಿರಾಂಕ std::vector<uint32_t> pc_cpy_fp16 =
            { (uint32_t)q->ne[0], (uint32_t)q->ne[1], (uint32_t)q->ne[2], (uint32_t)q->ne[3], (uint32_t)(x_ne) };
            ggml_vk_sync_buffers(ctx, subctx);
            ggml_pipeline_request_descriptor_sets(ctx, ಗಮ್ಯ_fp16_vk_0, 1);
            ggml_vk_dispatch_pipeline(ctx, subctx, ಗಮ್ಯ_fp16_vk_0, { q_buf, q_ತಾತ್ಕಾಲಿಕ_buf }, pc_cpy_fp16, { (uint32_t)(x_ne), 1, 1 });
        }

        ggml_vk_sync_buffers(ctx, subctx);
        ggml_pipeline_request_descriptor_sets(ctx, xe_fa_pipeline_dual_phases.ಮೊದಲ, 1);
        ggml_vk_dispatch_pipeline(ctx, subctx, xe_fa_pipeline_dual_phases.ಮೊದಲ,
            { q_ತಾತ್ಕಾಲಿಕ_buf, k_buf, ಮುಖವಾಡ_buf, p_ತಾತ್ಕಾಲಿಕ_buf, max_ತಾತ್ಕಾಲಿಕ_buf },
            pc_ph1, { (uint32_t)ph1_wg, (uint32_t)nek1, (uint32_t)neq3 });

        ggml_vk_sync_buffers(ctx, subctx);
        ggml_pipeline_request_descriptor_sets(ctx, xe_fa_pipeline_dual_phases.second, 1);
        ggml_vk_dispatch_pipeline(ctx, subctx, xe_fa_pipeline_dual_phases.second,
            { p_ತಾತ್ಕಾಲಿಕ_buf, v_buf, max_ತಾತ್ಕಾಲಿಕ_buf, sinks_buf, dst_buf },
            pc_ph2, { (uint32_t)ph2_wg, (uint32_t)nev2, (uint32_t)neq3 });

        ctx->prealloc_x_need_sync = ನಿಜ;
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (ವಿಭಜಿಸು_k > 1) {
        ggml_pipeline_request_descriptor_sets(ctx, ctx->device->pipeline_flash_attn_ವಿಭಜಿಸು_k_reduce, 1);

        ಆದರೆ (ctx->prealloc_ವಿಭಜಿಸು_k_need_sync) {
            ggml_vk_sync_buffers(ctx, subctx);
        }

        // We reuse workgroups_x to mean the number of splits, so we need to
        // cancel out the divide by wg_denoms[0].
        uint32_t dispatch_x;
        ಆದರೆ (gqa_ಅನುಪಾತ > 1) {
            workgroups_x *= pipeline->wg_denoms[0];
            dispatch_x = ವಿಭಜಿಸು_k * workgroups_x;
        } ಇಲ್ಲದಿದ್ದರೆ {
            dispatch_x = Tr * ವಿಭಜಿಸು_k * pipeline->wg_denoms[0];
        }

        vk_subbuffer ವಿಭಜಿಸು_k_buf = ggml_vk_subbuffer(ctx, ctx->prealloc_ವಿಭಜಿಸು_k, 0);
        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline,
                                    {q_buf, k_buf, v_buf, ಮುಖವಾಡ_buf, sinks_buf, ವಿಭಜಿಸು_k_buf, ಮುಖವಾಡ_opt_buf, sparse_buf},
                                    pc, { dispatch_x, workgroups_y, workgroups_z });

        ggml_vk_sync_buffers(ctx, subctx);
        ಸ್ಥಿರಾಂಕ vk_op_flash_attn_ವಿಭಜಿಸು_k_reduce_ತಳ್ಳು_constants pc2 = { HSV, (uint32_t)ne1, (uint32_t)ne2, (uint32_t)ne3, ವಿಭಜಿಸು_k, (sinks != ಶೂನ್ಯಸೂಚಕ) };
        ggml_vk_dispatch_pipeline(ctx, subctx, ctx->device->pipeline_flash_attn_ವಿಭಜಿಸು_k_reduce,
                                    {ವಿಭಜಿಸು_k_buf, sinks_buf, dst_buf},
                                    pc2, { (uint32_t)ne1, HSV, (uint32_t)(ne2 * ne3) });
        ctx->prealloc_ವಿಭಜಿಸು_k_need_sync = ನಿಜ;
    } ಇಲ್ಲದಿದ್ದರೆ {
        ಆದರೆ (gqa_ಅನುಪಾತ > 1) {
            // When using gqa, we want one actual workgroup per batch, so cancel out wg_denoms
            workgroups_x *= pipeline->wg_denoms[0];
        }
        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline,
                                    {q_buf, k_buf, v_buf, ಮುಖವಾಡ_buf, sinks_buf, dst_buf, ಮುಖವಾಡ_opt_buf, sparse_buf},
                                    pc, { workgroups_x, workgroups_y, workgroups_z });
    }

    ಆದರೆ (use_dequant_kv) {
        ctx->prealloc_x_need_sync = ನಿಜ;
    }
    ಆದರೆ (use_ಮುಖವಾಡ_opt || use_sparse) {
        ctx->prealloc_y_need_sync = ನಿಜ;
    }
}

ಸ್ಥಿರ vk_conv_shapes ggml_vk_conv_select_shape(ggml_backend_vk_context * ctx, uint32_t K, uint32_t NPQ) {
    ಸ್ವಯಂಚಾಲಿತ n_tiles = [&](vk_conv_shapes s) {
        ಹಿಂತಿರುಗಿ CEIL_DIV(K, vk_conv_ಖಂಡ_sizes[s].K)
            * CEIL_DIV(NPQ, vk_conv_ಖಂಡ_sizes[s].NPQ);
    };

    // We can't query number of shader cores on Intel, use 32 as a placeholder
    // so small convolutions will still choose a smaller tile.
    ಸ್ಥಿರಾಂಕ uint32_t shader_core_ಎಣಿಕೆ = ctx->device->shader_core_ಎಣಿಕೆ > 0 ? ctx->device->shader_core_ಎಣಿಕೆ : 32;

    // 128x128 isn't used with cm1 due to shared memory size; fall through to a smaller tile.
    ತಾರ್ಕಿಕ allow_128x128 = ನಿಜ;
#ಆದರೆ defined(VK_KHR_cooperative_ಮಾತೃಕೆ) && defined(GGML_VULKAN_COOPMAT_GLSLC_SUPPORT)
    ಆದರೆ (!ctx->device->coopmat2 && ctx->device->coopmat_support && ctx->device->coopmat_support_16x16x16_f16acc) {
        allow_128x128 = ಸುಳ್ಳು;
    }
#ಆದರೆ_ಅಂತ್ಯ

    ಆದರೆ (allow_128x128 && K > 64 && n_tiles(CONV_SHAPE_128x128) >= shader_core_ಎಣಿಕೆ * 2) {
        ಹಿಂತಿರುಗಿ CONV_SHAPE_128x128;
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (K <= 32 && n_tiles(CONV_SHAPE_32x256) >= shader_core_ಎಣಿಕೆ * 2) {
        ಹಿಂತಿರುಗಿ CONV_SHAPE_32x256;
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (K <= 64 && n_tiles(CONV_SHAPE_64x128) >= shader_core_ಎಣಿಕೆ * 2) {
        ಹಿಂತಿರುಗಿ CONV_SHAPE_64x128;
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (!allow_128x128 && K > 64 && n_tiles(CONV_SHAPE_64x128) >= shader_core_ಎಣಿಕೆ * 2) {
        // cm1 fallback for large K when 128x128 isn't available
        ಹಿಂತಿರುಗಿ CONV_SHAPE_64x128;
    } ಇಲ್ಲದಿದ್ದರೆ {
        ಹಿಂತಿರುಗಿ CONV_SHAPE_64x32;
    }
}

ಸ್ಥಿರ vk_pipeline ggml_vk_op_get_pipeline(ggml_backend_vk_context * ctx, ಸ್ಥಿರಾಂಕ ggml_tensor * src0, ಸ್ಥಿರಾಂಕ ggml_tensor * src1, ಸ್ಥಿರಾಂಕ ggml_tensor * src2, ಸ್ಥಿರಾಂಕ ggml_tensor * dst, ggml_op op) {
    ಆಯ್ಕೆ (op) {
    ಸ್ಥಿತಿ GGML_OP_GET_ಅಡ್ಡಸಾಲುಗಳು:
        GGML_ASSERT(src1->type == GGML_TYPE_I32);
        ಆದರೆ (src0->type == GGML_TYPE_I32) {
            // i32 src only supports i32 result
            GGML_ASSERT(dst->type == GGML_TYPE_I32);
            ಹಿಂತಿರುಗಿ ctx->device->pipeline_get_ಅಡ್ಡಸಾಲುಗಳು[src0->type];
        }
        ಆದರೆ (dst->type == GGML_TYPE_F16) {
            ಹಿಂತಿರುಗಿ ctx->device->pipeline_get_ಅಡ್ಡಸಾಲುಗಳು[src0->type];
        }
        ಆದರೆ (dst->type == GGML_TYPE_F32) {
            ಹಿಂತಿರುಗಿ ctx->device->pipeline_get_ಅಡ್ಡಸಾಲುಗಳು_f32[src0->type];
        }
        ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
    ಸ್ಥಿತಿ GGML_OP_GET_ಅಡ್ಡಸಾಲುಗಳು_ಹಿಂಭಾಗ:
        ಆದರೆ (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_I32 && dst->type == GGML_TYPE_F32) {
            ಹಿಂತಿರುಗಿ ctx->device->pipeline_get_ಅಡ್ಡಸಾಲುಗಳು_ಹಿಂಭಾಗ_f32;
        }
        ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
    ಸ್ಥಿತಿ GGML_OP_ACC:
        ಆದರೆ (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            ಹಿಂತಿರುಗಿ ctx->device->pipeline_acc_f32;
        }
        ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
    ಸ್ಥಿತಿ GGML_OP_ಗಣ:
        ಆದರೆ (src0->type == src1->type && src0->type == dst->type &&
            (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_I32)) {
            ಹಿಂತಿರುಗಿ ctx->device->pipeline_ಗಣ_f32;
        }
        ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
    ಸ್ಥಿತಿ GGML_OP_ಕೂಡಿಸು:
    ಸ್ಥಿತಿ GGML_OP_SUB:
    ಸ್ಥಿತಿ GGML_OP_MUL:
    ಸ್ಥಿತಿ GGML_OP_DIV:
        ಆದರೆ ((src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_F16) ||
            (src1->type != GGML_TYPE_F32 && src1->type != GGML_TYPE_F16) ||
            (dst->type != GGML_TYPE_F32 && dst->type != GGML_TYPE_F16)) {
            ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
        }
        ಆಯ್ಕೆ (op) {
        ಸ್ಥಿತಿ GGML_OP_ಕೂಡಿಸು:
        {
            ಆದರೆ (ctx->ಅಂಕೆ_additional_fused_ops > 0) {
                ಆದರೆ (ctx->do_ಕೂಡಿಸು_rms_partials) {
                    ಹಿಂತಿರುಗಿ ctx->device->pipeline_multi_ಕೂಡಿಸು_rms[ctx->ಅಂಕೆ_additional_fused_ops];
                } ಇಲ್ಲದಿದ್ದರೆ {
                    ಹಿಂತಿರುಗಿ ctx->device->pipeline_multi_ಕೂಡಿಸು[ctx->ಅಂಕೆ_additional_fused_ops];
                }
            }
            ಆದರೆ (ctx->do_ಕೂಡಿಸು_rms_partials) {
                ಸ್ವಯಂಚಾಲಿತ pipelines = ggml_are_same_shape(src0, src1) ? ctx->device->pipeline_ಕೂಡಿಸು_rms_norepeat : ctx->device->pipeline_ಕೂಡಿಸು_rms;
                ಹಿಂತಿರುಗಿ pipelines[src0->type == GGML_TYPE_F16][src1->type == GGML_TYPE_F16][dst->type == GGML_TYPE_F16];
            } ಇಲ್ಲದಿದ್ದರೆ {
                ಸ್ವಯಂಚಾಲಿತ pipelines = ggml_are_same_shape(src0, src1) ? ctx->device->pipeline_ಕೂಡಿಸು_norepeat : ctx->device->pipeline_ಕೂಡಿಸು;
                ಹಿಂತಿರುಗಿ pipelines[src0->type == GGML_TYPE_F16][src1->type == GGML_TYPE_F16][dst->type == GGML_TYPE_F16];
            }
        }
        ಸ್ಥಿತಿ GGML_OP_SUB:
        {
            ಸ್ವಯಂಚಾಲಿತ pipelines = ggml_are_same_shape(src0, src1) ? ctx->device->pipeline_sub_norepeat : ctx->device->pipeline_sub;
            ಹಿಂತಿರುಗಿ pipelines[src0->type == GGML_TYPE_F16][src1->type == GGML_TYPE_F16][dst->type == GGML_TYPE_F16];
        }
        ಸ್ಥಿತಿ GGML_OP_MUL:
        {
            ಸ್ವಯಂಚಾಲಿತ pipelines = ggml_are_same_shape(src0, src1) ? ctx->device->pipeline_mul_norepeat : ctx->device->pipeline_mul;
            ಹಿಂತಿರುಗಿ pipelines[src0->type == GGML_TYPE_F16][src1->type == GGML_TYPE_F16][dst->type == GGML_TYPE_F16];
        }
        ಸ್ಥಿತಿ GGML_OP_DIV:
        {
            ಸ್ವಯಂಚಾಲಿತ pipelines = ggml_are_same_shape(src0, src1) ? ctx->device->pipeline_div_norepeat : ctx->device->pipeline_div;
            ಹಿಂತಿರುಗಿ pipelines[src0->type == GGML_TYPE_F16][src1->type == GGML_TYPE_F16][dst->type == GGML_TYPE_F16];
        }
        ಪೂರ್ವನಿಯೋಜಿತ:
            ನಿಲ್ಲಿಸು;
        }
        ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
    ಸ್ಥಿತಿ GGML_OP_ಕೂಡಿಸು_ID:
        ಆದರೆ (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 && src2->type == GGML_TYPE_I32 && dst->type == GGML_TYPE_F32) {
            ಹಿಂತಿರುಗಿ ctx->device->pipeline_ಕೂಡಿಸು_id_f32;
        }
        ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
    ಸ್ಥಿತಿ GGML_OP_ಹೊರಹೋಗು_PROD:
        ಆದರೆ (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            ಹಿಂತಿರುಗಿ ctx->device->pipeline_ಹೊರಹೋಗು_prod_f32;
        }
        ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
    ಸ್ಥಿತಿ GGML_OP_CONCAT: {
        ಆದರೆ (!ggml_vk_concat_supported(src0, src1, dst)) {
            ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
        }
        ಆಯ್ಕೆ (ggml_vk_concat_unit_ಗಾತ್ರ(src0->type)) {
        ಸ್ಥಿತಿ 1:
            ಹಿಂತಿರುಗಿ ctx->device->pipeline_concat_i8;
        ಸ್ಥಿತಿ 2:
            ಹಿಂತಿರುಗಿ ctx->device->pipeline_concat_i16;
        ಸ್ಥಿತಿ 4:
            ಹಿಂತಿರುಗಿ ctx->device->pipeline_concat_i32;
        ಸ್ಥಿತಿ 8:
            ಹಿಂತಿರುಗಿ ctx->device->pipeline_concat_i64;
        ಪೂರ್ವನಿಯೋಜಿತ:
            ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
        }
    }
    ಸ್ಥಿತಿ GGML_OP_UPSCALE:
        ಆದರೆ (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            uint32_t mode = (ggml_get_op_params_i32(dst, 0) & (0xFF | GGML_SCALE_FLAG_ANTIALIAS));
            ಆಯ್ಕೆ (mode) {
                ಸ್ಥಿತಿ GGML_SCALE_ಬಹುಲಕ_ಹತ್ತಿರದ:
                    ಹಿಂತಿರುಗಿ ctx->device->pipeline_upscale_ಹತ್ತಿರದ_f32;
                ಸ್ಥಿತಿ GGML_SCALE_ಬಹುಲಕ_BILINEAR:
                    ಹಿಂತಿರುಗಿ ctx->device->pipeline_upscale_bilinear_f32;
                ಸ್ಥಿತಿ GGML_SCALE_ಬಹುಲಕ_BICUBIC:
                    ಹಿಂತಿರುಗಿ ctx->device->pipeline_upscale_bicubic_f32;
                ಸ್ಥಿತಿ GGML_SCALE_ಬಹುಲಕ_BILINEAR | GGML_SCALE_FLAG_ANTIALIAS:
                    ಹಿಂತಿರುಗಿ ctx->device->pipeline_upscale_bilinear_antialias_f32;
                ಪೂರ್ವನಿಯೋಜಿತ:
                    ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
            }
        }
        ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
    ಸ್ಥಿತಿ GGML_OP_SCALE:
        ಆದರೆ (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            ಹಿಂತಿರುಗಿ ctx->device->pipeline_scale_f32;
        }
        ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
    ಸ್ಥಿತಿ GGML_OP_SQR:
        ಆದರೆ (src0->type == dst->type &&
            (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16)) {
            ಹಿಂತಿರುಗಿ ctx->device->pipeline_sqr[dst->type == GGML_TYPE_F16];
        }
        ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
    ಸ್ಥಿತಿ GGML_OP_SQRT:
        ಆದರೆ (src0->type == dst->type &&
            (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16)) {
            ಹಿಂತಿರುಗಿ ctx->device->pipeline_sqrt[dst->type == GGML_TYPE_F16];
        }
        ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
    ಸ್ಥಿತಿ GGML_OP_SIN:
        ಆದರೆ (src0->type == dst->type &&
            (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16)) {
            ಹಿಂತಿರುಗಿ ctx->device->pipeline_sin[dst->type == GGML_TYPE_F16];
        }
        ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
    ಸ್ಥಿತಿ GGML_OP_COS:
        ಆದರೆ (src0->type == dst->type &&
            (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16)) {
            ಹಿಂತಿರುಗಿ ctx->device->pipeline_cos[dst->type == GGML_TYPE_F16];
        }
        ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
    ಸ್ಥಿತಿ GGML_OP_LOG:
        ಆದರೆ (src0->type == dst->type &&
            (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16)) {
            ಹಿಂತಿರುಗಿ ctx->device->pipeline_log[dst->type == GGML_TYPE_F16];
        }
        ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
    ಸ್ಥಿತಿ GGML_OP_TRI:
        ಆದರೆ (src0->type == dst->type &&
            (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16)) {
            ಹಿಂತಿರುಗಿ ctx->device->pipeline_tri[dst->type == GGML_TYPE_F16];
        }
        ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
    ಸ್ಥಿತಿ GGML_OP_DIAG:
        ಆದರೆ (src0->type == dst->type &&
            (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16)) {
            ಹಿಂತಿರುಗಿ ctx->device->pipeline_diag[dst->type == GGML_TYPE_F16];
        }
        ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
    ಸ್ಥಿತಿ GGML_OP_CLAMP:
        ಆದರೆ (src0->type == dst->type &&
            (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16)) {
            ಹಿಂತಿರುಗಿ ctx->device->pipeline_clamp[dst->type == GGML_TYPE_F16];
        }
        ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
    ಸ್ಥಿತಿ GGML_OP_PAD:
        ಆದರೆ (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            ಹಿಂತಿರುಗಿ ctx->device->pipeline_pad_f32;
        }
        ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
    ಸ್ಥಿತಿ GGML_OP_PAD_REFLECT_1D:
        ಆದರೆ (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            ಹಿಂತಿರುಗಿ ctx->device->pipeline_pad_reflect_1d_f32;
        }
        ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
    ಸ್ಥಿತಿ GGML_OP_ROLL:
        ಆದರೆ (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            ಹಿಂತಿರುಗಿ ctx->device->pipeline_roll_f32;
        }
        ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
    ಸ್ಥಿತಿ GGML_OP_REPEAT:
        ಆದರೆ (ggml_type_ಗಾತ್ರ(src0->type) == ಗಾತ್ರ(ದಶಮಾಂಶ) && ggml_type_ಗಾತ್ರ(dst->type) == ಗಾತ್ರ(ದಶಮಾಂಶ)) {
            ಹಿಂತಿರುಗಿ ctx->device->pipeline_repeat_i32;
        }
        ಆದರೆ (ggml_type_ಗಾತ್ರ(src0->type) == 2 && ggml_type_ಗಾತ್ರ(dst->type) == 2) {
            ಹಿಂತಿರುಗಿ ctx->device->pipeline_repeat_i16;
        }
        ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
    ಸ್ಥಿತಿ GGML_OP_REPEAT_ಹಿಂಭಾಗ:
        ಆದರೆ (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            ಹಿಂತಿರುಗಿ ctx->device->pipeline_repeat_ಹಿಂಭಾಗ_f32;
        }
        ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
    ಸ್ಥಿತಿ GGML_OP_CPY:
    ಸ್ಥಿತಿ GGML_OP_CONT:
    ಸ್ಥಿತಿ GGML_OP_DUP:
        ಹಿಂತಿರುಗಿ ggml_vk_get_cpy_pipeline(ctx, src0, dst, dst->type);
    ಸ್ಥಿತಿ GGML_OP_ಗಣ_ಅಡ್ಡಸಾಲುಗಳು:
        {
            ಆದರೆ (src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_F16) {
                ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
            }
            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ src_idx = src0->type == GGML_TYPE_F16;
            ಆದರೆ (src1->type == GGML_TYPE_I64) {
                ಹಿಂತಿರುಗಿ ctx->device->pipeline_ಗಣ_ಅಡ್ಡಸಾಲುಗಳು_i64[src_idx][dst->type];
            } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (src1->type == GGML_TYPE_I32) {
                ಹಿಂತಿರುಗಿ ctx->device->pipeline_ಗಣ_ಅಡ್ಡಸಾಲುಗಳು_i32[src_idx][dst->type];
            }
            ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
        }
    ಸ್ಥಿತಿ GGML_OP_SILU_ಹಿಂಭಾಗ:
        ಆದರೆ (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            ಹಿಂತಿರುಗಿ ctx->device->pipeline_silu_ಹಿಂಭಾಗ_f32;
        }
        ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
    ಸ್ಥಿತಿ GGML_OP_NORM:
        ಆದರೆ (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            ಹಿಂತಿರುಗಿ ctx->device->pipeline_norm_f32;
        }
        ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
    ಸ್ಥಿತಿ GGML_OP_ಗುಂಪು_NORM:
        ಆದರೆ (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            ಹಿಂತಿರುಗಿ ctx->device->pipeline_ಗುಂಪು_norm_f32;
        }
        ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
    ಸ್ಥಿತಿ GGML_OP_RMS_NORM:
        ಆದರೆ (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            ಆದರೆ (ctx->do_ಕೂಡಿಸು_rms_partials) {
                ಹಿಂತಿರುಗಿ ctx->fused_rms_norm_ಬಹುಲಕ == RMS_NORM_MUL ? ctx->device->pipeline_rms_norm_mul_partials_f32 : ctx->device->pipeline_rms_norm_partials_f32;
            }
            ಹಿಂತಿರುಗಿ ctx->fused_rms_norm_ಬಹುಲಕ == RMS_NORM_MUL ? ctx->device->pipeline_rms_norm_mul_f32 : ctx->device->pipeline_rms_norm_f32;
        }
        ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
    ಸ್ಥಿತಿ GGML_OP_RMS_NORM_ಹಿಂಭಾಗ:
        ಆದರೆ (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            ಹಿಂತಿರುಗಿ ctx->device->pipeline_rms_norm_ಹಿಂಭಾಗ_f32;
        }
        ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
    ಸ್ಥಿತಿ GGML_OP_L2_NORM:
        ಆದರೆ (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            ಹಿಂತಿರುಗಿ ctx->device->pipeline_l2_norm_f32;
        }
        ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
    ಸ್ಥಿತಿ GGML_OP_UNARY:
        ಆದರೆ ((src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_F16) ||
            (dst->type != GGML_TYPE_F32 && dst->type != GGML_TYPE_F16) ||
            (src0->type != dst->type)) {
            ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
        }

        ಆಯ್ಕೆ (ggml_get_unary_op(dst)) {
            ಸ್ಥಿತಿ GGML_UNARY_OP_EXP:
                ಹಿಂತಿರುಗಿ ctx->device->pipeline_exp[dst->type == GGML_TYPE_F16];
            ಸ್ಥಿತಿ GGML_UNARY_OP_EXPM1:
                ಹಿಂತಿರುಗಿ ctx->device->pipeline_expm1[dst->type == GGML_TYPE_F16];
            ಸ್ಥಿತಿ GGML_UNARY_OP_ELU:
                ಹಿಂತಿರುಗಿ ctx->device->pipeline_elu[dst->type == GGML_TYPE_F16];
            ಸ್ಥಿತಿ GGML_UNARY_OP_SILU:
                ಹಿಂತಿರುಗಿ ctx->device->pipeline_silu[dst->type == GGML_TYPE_F16];
            ಸ್ಥಿತಿ GGML_UNARY_OP_GELU:
                ಹಿಂತಿರುಗಿ ctx->device->pipeline_gelu[dst->type == GGML_TYPE_F16];
            ಸ್ಥಿತಿ GGML_UNARY_OP_GELU_ERF:
                ಹಿಂತಿರುಗಿ ctx->device->pipeline_gelu_erf[dst->type == GGML_TYPE_F16];
            ಸ್ಥಿತಿ GGML_UNARY_OP_GELU_ಶೀಘ್ರ:
                ಹಿಂತಿರುಗಿ ctx->device->pipeline_gelu_ಶೀಘ್ರ[dst->type == GGML_TYPE_F16];
            ಸ್ಥಿತಿ GGML_UNARY_OP_RELU:
                ಹಿಂತಿರುಗಿ ctx->device->pipeline_relu[dst->type == GGML_TYPE_F16];
            ಸ್ಥಿತಿ GGML_UNARY_OP_XIELU:
                ಹಿಂತಿರುಗಿ ctx->device->pipeline_xielu[dst->type == GGML_TYPE_F16];
            ಸ್ಥಿತಿ GGML_UNARY_OP_NEG:
                ಹಿಂತಿರುಗಿ ctx->device->pipeline_neg[dst->type == GGML_TYPE_F16];
            ಸ್ಥಿತಿ GGML_UNARY_OP_TANH:
                ಹಿಂತಿರುಗಿ ctx->device->pipeline_tanh[dst->type == GGML_TYPE_F16];
            ಸ್ಥಿತಿ GGML_UNARY_OP_SIGMOID:
                ಹಿಂತಿರುಗಿ ctx->device->pipeline_sigmoid[dst->type == GGML_TYPE_F16];
            ಸ್ಥಿತಿ GGML_UNARY_OP_HARDSIGMOID:
                ಹಿಂತಿರುಗಿ ctx->device->pipeline_hardsigmoid[dst->type == GGML_TYPE_F16];
            ಸ್ಥಿತಿ GGML_UNARY_OP_HARDSWISH:
                ಹಿಂತಿರುಗಿ ctx->device->pipeline_hardswish[dst->type == GGML_TYPE_F16];
            ಸ್ಥಿತಿ GGML_UNARY_OP_ABS:
                ಹಿಂತಿರುಗಿ ctx->device->pipeline_abs[dst->type == GGML_TYPE_F16];
            ಸ್ಥಿತಿ GGML_UNARY_OP_SOFTPLUS:
                ಹಿಂತಿರುಗಿ ctx->device->pipeline_softplus[dst->type == GGML_TYPE_F16];
            ಸ್ಥಿತಿ GGML_UNARY_OP_ಹೆಜ್ಜೆ:
                ಹಿಂತಿರುಗಿ ctx->device->pipeline_ಹೆಜ್ಜೆ[dst->type == GGML_TYPE_F16];
            ಸ್ಥಿತಿ GGML_UNARY_OP_ಸುತ್ತು:
                ಹಿಂತಿರುಗಿ ctx->device->pipeline_ಸುತ್ತು[dst->type == GGML_TYPE_F16];
            ಸ್ಥಿತಿ GGML_UNARY_OP_CEIL:
                ಹಿಂತಿರುಗಿ ctx->device->pipeline_ceil[dst->type == GGML_TYPE_F16];
            ಸ್ಥಿತಿ GGML_UNARY_OP_FLOOR:
                ಹಿಂತಿರುಗಿ ctx->device->pipeline_floor[dst->type == GGML_TYPE_F16];
            ಸ್ಥಿತಿ GGML_UNARY_OP_TRUNC:
                ಹಿಂತಿರುಗಿ ctx->device->pipeline_trunc[dst->type == GGML_TYPE_F16];
            ಸ್ಥಿತಿ GGML_UNARY_OP_SGN:
                ಹಿಂತಿರುಗಿ ctx->device->pipeline_sgn[dst->type == GGML_TYPE_F16];
            ಪೂರ್ವನಿಯೋಜಿತ:
                ನಿಲ್ಲಿಸು;
        }
        ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
    ಸ್ಥಿತಿ GGML_OP_GLU:
        ಆದರೆ ((src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_F16) ||
            (dst->type != GGML_TYPE_F32 && dst->type != GGML_TYPE_F16) ||
            (src0->type != dst->type)) {
            ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
        }

        ಆಯ್ಕೆ (ggml_get_glu_op(dst)) {
            ಸ್ಥಿತಿ GGML_GLU_OP_GEGLU:
                ಹಿಂತಿರುಗಿ ctx->device->pipeline_geglu[dst->type == GGML_TYPE_F16];
            ಸ್ಥಿತಿ GGML_GLU_OP_REGLU:
                ಹಿಂತಿರುಗಿ ctx->device->pipeline_reglu[dst->type == GGML_TYPE_F16];
            ಸ್ಥಿತಿ GGML_GLU_OP_SWIGLU:
                ಹಿಂತಿರುಗಿ ctx->device->pipeline_swiglu[dst->type == GGML_TYPE_F16];
            ಸ್ಥಿತಿ GGML_GLU_OP_SWIGLU_OAI:
                ಹಿಂತಿರುಗಿ ctx->device->pipeline_swiglu_oai[dst->type == GGML_TYPE_F16];
            ಸ್ಥಿತಿ GGML_GLU_OP_SWIGLU_CLAMP:
                ಹಿಂತಿರುಗಿ ctx->device->pipeline_swiglu_clamp[dst->type == GGML_TYPE_F16];
            ಸ್ಥಿತಿ GGML_GLU_OP_GEGLU_ERF:
                ಹಿಂತಿರುಗಿ ctx->device->pipeline_geglu_erf[dst->type == GGML_TYPE_F16];
            ಸ್ಥಿತಿ GGML_GLU_OP_GEGLU_ಶೀಘ್ರ:
                ಹಿಂತಿರುಗಿ ctx->device->pipeline_geglu_ಶೀಘ್ರ[dst->type == GGML_TYPE_F16];
            ಪೂರ್ವನಿಯೋಜಿತ:
                ನಿಲ್ಲಿಸು;
        }
        ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
    ಸ್ಥಿತಿ GGML_OP_DIAG_ಮುಖವಾಡ_INF:
        ಆದರೆ (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            ಹಿಂತಿರುಗಿ ctx->device->pipeline_diag_ಮುಖವಾಡ_inf_f32;
        }
        ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
    ಸ್ಥಿತಿ GGML_OP_SOFT_MAX:
        GGML_ASSERT(!src1 || src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16);
        GGML_ASSERT(!src2 || src2->type == GGML_TYPE_F32);

        ಆದರೆ (ctx->ಅಂಕೆ_additional_fused_ops) {
            uint32_t idx = (uint32_t)ceilf(log2f(ದಶಮಾಂಶ(dst->ne[0])));
            GGML_ASSERT(idx < ಅಂಕೆ_topk_moe_pipelines);
            // use n_experts from push constant if it's not equal to the power of two spec constant
            ತಾರ್ಕಿಕ use_ತಳ್ಳು = dst->ne[0] != (1u << idx);
            ಹಿಂತಿರುಗಿ ctx->device->pipeline_topk_moe[idx][use_ತಳ್ಳು];
        }

        ಆದರೆ (src0->type == GGML_TYPE_F32 && (src1 == ಶೂನ್ಯಸೂಚಕ || src1->type == GGML_TYPE_F32) && dst->type == GGML_TYPE_F32) {
            ಹಿಂತಿರುಗಿ src0->ne[0] > 1024 ? ctx->device->pipeline_soft_max_f32_wg512 : ctx->device->pipeline_soft_max_f32;
        }
        ಆದರೆ (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F32) {
            ಹಿಂತಿರುಗಿ src0->ne[0] > 1024 ? ctx->device->pipeline_soft_max_f32_f16_wg512 : ctx->device->pipeline_soft_max_f32_f16;
        }
        ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
    ಸ್ಥಿತಿ GGML_OP_SOFT_MAX_ಹಿಂಭಾಗ:
        ಆದರೆ (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            ಹಿಂತಿರುಗಿ ctx->device->pipeline_soft_max_ಹಿಂಭಾಗ_f32;
        }
        ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
    ಸ್ಥಿತಿ GGML_OP_ROPE:
    ಸ್ಥಿತಿ GGML_OP_ROPE_ಹಿಂಭಾಗ:
        {
            ಸ್ಥಿರಾಂಕ ggml_tensor *rope = ctx->ಅಂಕೆ_additional_fused_ops == 2 ? dst->src[0]->src[0] : dst;
            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ mode = ((ಸ್ಥಿರಾಂಕ int32_t *) rope->op_params)[2];
            ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ is_neox = mode & GGML_ROPE_TYPE_NEOX;
            ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ is_mrope = mode & GGML_ROPE_TYPE_MROPE;
            ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ is_vision = mode == GGML_ROPE_TYPE_VISION;

            ಆದರೆ (is_neox) {
                ಆದರೆ (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
                    ಹಿಂತಿರುಗಿ ctx->device->pipeline_rope_neox_f32;
                }
                ಆದರೆ (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F16) {
                    ಹಿಂತಿರುಗಿ ctx->device->pipeline_rope_neox_f32_f16;
                }
                ಆದರೆ (src0->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F16) {
                    ಹಿಂತಿರುಗಿ ctx->device->pipeline_rope_neox_f16;
                }
            } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (is_mrope && !is_vision) {
                ಆದರೆ (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
                    ಹಿಂತಿರುಗಿ ctx->device->pipeline_rope_multi_f32;
                }
                ಆದರೆ (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F16) {
                    ಹಿಂತಿರುಗಿ ctx->device->pipeline_rope_multi_f32_f16;
                }
                ಆದರೆ (src0->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F16) {
                    ಹಿಂತಿರುಗಿ ctx->device->pipeline_rope_multi_f16;
                }
            } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (is_vision) {
                ಆದರೆ (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
                    ಹಿಂತಿರುಗಿ ctx->device->pipeline_rope_vision_f32;
                }
                ಆದರೆ (src0->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F16) {
                    ಹಿಂತಿರುಗಿ ctx->device->pipeline_rope_vision_f16;
                }
            } ಇಲ್ಲದಿದ್ದರೆ {
                ಆದರೆ (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
                    ಹಿಂತಿರುಗಿ ctx->device->pipeline_rope_norm_f32;
                }
                ಆದರೆ (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F16) {
                    ಹಿಂತಿರುಗಿ ctx->device->pipeline_rope_norm_f32_f16;
                }
                ಆದರೆ (src0->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F16) {
                    ಹಿಂತಿರುಗಿ ctx->device->pipeline_rope_norm_f16;
                }
            }
            ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
        }
    ಸ್ಥಿತಿ GGML_OP_ಮೊತ್ತ:
    ಸ್ಥಿತಿ GGML_OP_ಮೊತ್ತ_ಅಡ್ಡಸಾಲುಗಳು:
    ಸ್ಥಿತಿ GGML_OP_ಸರಾಸರಿ:
        ಆದರೆ (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            ಹಿಂತಿರುಗಿ ctx->device->pipeline_ಮೊತ್ತ_ಅಡ್ಡಸಾಲುಗಳು_f32;
        }
        ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
    ಸ್ಥಿತಿ GGML_OP_CROSS_ENTROPY_LOSS:
        ಆದರೆ (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            ಹಿಂತಿರುಗಿ src0->ne[0] > 1024 ? ctx->device->pipeline_cross_entropy_loss_f32_wg512 : ctx->device->pipeline_cross_entropy_loss_f32;
        }
        ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
    ಸ್ಥಿತಿ GGML_OP_CROSS_ENTROPY_LOSS_ಹಿಂಭಾಗ:
        // src0 is the scalar grad; src1 is logits
        ಆದರೆ (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 && src2 && src2->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            ಹಿಂತಿರುಗಿ src1->ne[0] > 1024 ? ctx->device->pipeline_cross_entropy_loss_ಹಿಂಭಾಗ_f32_wg512 : ctx->device->pipeline_cross_entropy_loss_ಹಿಂಭಾಗ_f32;
        }
        ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
    ಸ್ಥಿತಿ GGML_OP_CUMSUM:
        ಆದರೆ (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            ಆದರೆ (src0->ne[0] <= 512) {
                ಹಿಂತಿರುಗಿ ctx->device->pipeline_cumsum_small_f32;
            } ಇಲ್ಲದಿದ್ದರೆ {
                ಹಿಂತಿರುಗಿ ctx->device->pipeline_cumsum_f32;
            }
        }
        ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
    ಸ್ಥಿತಿ GGML_OP_SOLVE_TRI:
        ಆದರೆ (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {

            vk_solve_tri_pipeline_state solve_tri_pipeline_state(src0->ne[0], src1->ne[0]);

            vk_pipeline pipeline = ಶೂನ್ಯಸೂಚಕ;

            {
                std::lock_guard<std::mutex> guard(ctx->device->compile_mutex);
                ಸ್ವಯಂಚಾಲಿತ it = ctx->device->pipeline_solve_tri_f32.find(solve_tri_pipeline_state);
                ಆದರೆ (it != ctx->device->pipeline_solve_tri_f32.end()) {
                    pipeline = it->second;
                } ಇಲ್ಲದಿದ್ದರೆ {
                    ctx->device->pipeline_solve_tri_f32[solve_tri_pipeline_state] = pipeline = std::make_shared<vk_pipeline_struct>();
                }
            }

            ಹಿಂತಿರುಗಿ pipeline;
        }
        ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
    ಸ್ಥಿತಿ GGML_OP_ARGMAX:
        ಆದರೆ (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_I32) {
            ಹಿಂತಿರುಗಿ ctx->device->pipeline_argmax_f32;
        }
        ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
    ಸ್ಥಿತಿ GGML_OP_ಎಣಿಕೆ_EQUAL:
        ಆದರೆ (src0->type == GGML_TYPE_I32 && src1->type == GGML_TYPE_I32 && dst->type == GGML_TYPE_I64) {
            ಹಿಂತಿರುಗಿ ctx->device->pipeline_ಎಣಿಕೆ_equal_i32;
        }
        ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
    ಸ್ಥಿತಿ GGML_OP_IM2_ಲಂಬಸಾಲು:
        ಆದರೆ (src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            ಹಿಂತಿರುಗಿ ctx->device->pipeline_im2col_f32;
        }
        ಆದರೆ (src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F16) {
            ಹಿಂತಿರುಗಿ ctx->device->pipeline_im2col_f32_f16;
        }
        ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
    ಸ್ಥಿತಿ GGML_OP_IM2_ಲಂಬಸಾಲು_3_D:
        ಆದರೆ (src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            ಹಿಂತಿರುಗಿ ctx->device->pipeline_im2col_3d_f32;
        }
        ಆದರೆ (src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F16) {
            ಹಿಂತಿರುಗಿ ctx->device->pipeline_im2col_3d_f32_f16;
        }
        ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
    ಸ್ಥಿತಿ GGML_OP_TIMESTEP_EMBEDDING:
        ಆದರೆ (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            ಹಿಂತಿರುಗಿ ctx->device->pipeline_timestep_embedding_f32;
        }
        ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
    ಸ್ಥಿತಿ GGML_OP_CONV_TRANSPOSE_1D:
        ಆದರೆ (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            ಹಿಂತಿರುಗಿ ctx->device->pipeline_conv_transpose_1d_f32;
        }
        ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
    ಸ್ಥಿತಿ GGML_OP_COL2IM_1D:
        ಆಯ್ಕೆ (src0->type) {
            ಸ್ಥಿತಿ GGML_TYPE_F32:  ಹಿಂತಿರುಗಿ ctx->device->pipeline_col2im_1d_f32;
            ಸ್ಥಿತಿ GGML_TYPE_F16:  ಹಿಂತಿರುಗಿ ctx->device->pipeline_col2im_1d_f16;
            ಸ್ಥಿತಿ GGML_TYPE_BF16: ಹಿಂತಿರುಗಿ ctx->device->pipeline_col2im_1d_bf16;
            ಪೂರ್ವನಿಯೋಜಿತ:             ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
        }
    ಸ್ಥಿತಿ GGML_OP_POOL_1D:
        ಆದರೆ (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            ಹಿಂತಿರುಗಿ ctx->device->pipeline_pool1d_f32;
        }
        ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
    ಸ್ಥಿತಿ GGML_OP_POOL_2D:
        ಆದರೆ (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            // Wide windows use neighboring subgroup lanes to read neighboring inputs.
            ಆದರೆ (ctx->device->pipeline_pool2d_subgroup_f32 && dst->op_params[1] >= int32_t(ctx->device->subgroup_ಗಾತ್ರ)) {
                ಹಿಂತಿರುಗಿ ctx->device->pipeline_pool2d_subgroup_f32;
            }
            ಹಿಂತಿರುಗಿ ctx->device->pipeline_pool2d_f32;
        }
        ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
    ಸ್ಥಿತಿ GGML_OP_RWKV_WKV6:
        ಆದರೆ (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            ಹಿಂತಿರುಗಿ ctx->device->pipeline_rwkv_wkv6_f32;
        }
        ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
    ಸ್ಥಿತಿ GGML_OP_RWKV_WKV7:
        ಆದರೆ (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            ಹಿಂತಿರುಗಿ ctx->device->pipeline_rwkv_wkv7_f32;
        }
        ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
    ಸ್ಥಿತಿ GGML_OP_GATED_ರೇಖೀಯ_ATTN:
        ಆದರೆ (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            ಹಿಂತಿರುಗಿ ctx->device->pipeline_gated_ರೇಖೀಯ_attn_f32;
        }
        ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
    ಸ್ಥಿತಿ GGML_OP_LIGHTNING_INDEXER:
        // only the k type selects a pipeline, the other types are fixed by ggml_lightning_indexer()
        ಆದರೆ (ggml_vk_lightning_indexer_k_type_supported(src1->type)) {
            ಹಿಂತಿರುಗಿ ctx->device->pipeline_lightning_indexer_f32[src1->type];
        }
        ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
    ಸ್ಥಿತಿ GGML_OP_GATED_DELTA_NET:
        ಆದರೆ (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            ಸ್ಥಿರಾಂಕ uint32_t S_v = dst->src[2]->ne[0];
            ಸ್ಥಿರಾಂಕ uint32_t kda = (dst->src[3]->ne[0] == (int64_t)S_v) ? 1 : 0;
            uint32_t si;
            ಆಯ್ಕೆ (S_v) {
                ಸ್ಥಿತಿ 16:  si = 0; ನಿಲ್ಲಿಸು;
                ಸ್ಥಿತಿ 32:  si = 1; ನಿಲ್ಲಿಸು;
                ಸ್ಥಿತಿ 64:  si = 2; ನಿಲ್ಲಿಸು;
                ಸ್ಥಿತಿ 128: si = 3; ನಿಲ್ಲಿಸು;
                ಪೂರ್ವನಿಯೋಜಿತ: ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
            }
            ಹಿಂತಿರುಗಿ ctx->device->pipeline_gated_delta_net[si][kda];
        }
        ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
    ಸ್ಥಿತಿ GGML_OP_SSM_ಪರಿಶೀಲನೆ:
        ಆದರೆ (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            ಸ್ಥಿರಾಂಕ uint32_t d_state = src0->ne[0];
            ಆದರೆ (d_state == 128) {
                ಹಿಂತಿರುಗಿ ctx->device->pipeline_ssm_ಪರಿಶೀಲನೆ_f32_d128;
            } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (d_state == 256) {
                ಹಿಂತಿರುಗಿ ctx->device->pipeline_ssm_ಪರಿಶೀಲನೆ_f32_d256;
            }
        }
        ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
    ಸ್ಥಿತಿ GGML_OP_SSM_CONV:
        ಆದರೆ (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            ಆಯ್ಕೆ (ctx->ಅಂಕೆ_additional_fused_ops) {
                ಸ್ಥಿತಿ 0:  ಹಿಂತಿರುಗಿ ctx->device->pipeline_ssm_conv_f32;
                ಸ್ಥಿತಿ 1:  ಹಿಂತಿರುಗಿ ctx->device->pipeline_ssm_conv_silu_f32;
                ಸ್ಥಿತಿ 2:  ಹಿಂತಿರುಗಿ ctx->device->pipeline_ssm_conv_bias_silu_f32;
                ಪೂರ್ವನಿಯೋಜಿತ: ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
            }
        }
        ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
    ಸ್ಥಿತಿ GGML_OP_OPT_ಹೆಜ್ಜೆ_ADAMW:
        ಆದರೆ (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            ಹಿಂತಿರುಗಿ ctx->device->pipeline_opt_ಹೆಜ್ಜೆ_adamw_f32;
        }
        ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
    ಸ್ಥಿತಿ GGML_OP_OPT_ಹೆಜ್ಜೆ_SGD:
        ಆದರೆ (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            ಹಿಂತಿರುಗಿ ctx->device->pipeline_opt_ಹೆಜ್ಜೆ_sgd_f32;
        }
        ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
    ಸ್ಥಿತಿ GGML_OP_LEAKY_RELU:
        ಆದರೆ (src0->type == dst->type &&
            (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16)) {
            ಹಿಂತಿರುಗಿ ctx->device->pipeline_leaky_relu[dst->type == GGML_TYPE_F16];
        }
        ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
    ಸ್ಥಿತಿ GGML_OP_CONV_2D:
    ಸ್ಥಿತಿ GGML_OP_CONV_TRANSPOSE_2D:
        ಆದರೆ (src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
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
            ಸ್ಥಿರಾಂಕ uint32_t Cin = (uint32_t)src1->ne[2];
            ಸ್ಥಿರಾಂಕ uint32_t CRS = Cin * KW * KH;
            ಸ್ಥಿರಾಂಕ uint32_t BS_K   = vk_conv_ಖಂಡ_sizes[shape].K;
            ಸ್ಥಿರಾಂಕ uint32_t BS_CRS = vk_conv_ಖಂಡ_sizes[shape].CRS;
            ಸ್ಥಿರಾಂಕ uint32_t BS_NPQ = vk_conv_ಖಂಡ_sizes[shape].NPQ;
            ಸ್ಥಿರಾಂಕ uint32_t aligned = ((K   % BS_K   == 0) &&
                                      (CRS % BS_CRS == 0) &&
                                      (NPQ % BS_NPQ == 0)) ? 1u : 0u;

            vk_conv2d_pipeline_state conv2d_pipeline_state(s0, s1, p0, p1, d0, d1, KW, KH, aligned);

            std::map<vk_conv2d_pipeline_state, vk_pipeline> *pipelines = ಶೂನ್ಯಸೂಚಕ;
            ಆದರೆ (op == GGML_OP_CONV_2D) {
                ಆದರೆ (src0->type == GGML_TYPE_F32) {
                    pipelines = &ctx->device->pipeline_conv2d_f32[shape];
                } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (src0->type == GGML_TYPE_F16) {
                    pipelines = &ctx->device->pipeline_conv2d_f16_f32[shape];
                }
            } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (op == GGML_OP_CONV_TRANSPOSE_2D) {
                ಆದರೆ (src0->type == GGML_TYPE_F32) {
                    pipelines = &ctx->device->pipeline_conv_transpose_2d_f32[shape];
                } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (src0->type == GGML_TYPE_F16) {
                    pipelines = &ctx->device->pipeline_conv_transpose_2d_f16_f32[shape];
                }
            }

            vk_pipeline pipeline = ಶೂನ್ಯಸೂಚಕ;

            {
                std::lock_guard<std::mutex> guard(ctx->device->compile_mutex);
                ಸ್ವಯಂಚಾಲಿತ it = pipelines->find(conv2d_pipeline_state);
                ಆದರೆ (it != pipelines->end()) {
                    pipeline = it->second;
                } ಇಲ್ಲದಿದ್ದರೆ {
                    (*pipelines)[conv2d_pipeline_state] = pipeline = std::make_shared<vk_pipeline_struct>();
                }
            }

            ಹಿಂತಿರುಗಿ pipeline;
        }
        ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
    ಸ್ಥಿತಿ GGML_OP_CONV_2D_DW:
        ಆದರೆ (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            ಆದರೆ (ggml_is_contiguous(src1)) {
                ಹಿಂತಿರುಗಿ ctx->device->pipeline_conv2d_dw_whcn_f32;
            } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (ggml_is_contiguous_channels(src1)) {
                ಹಿಂತಿರುಗಿ ctx->device->pipeline_conv2d_dw_cwhn_f32;
            }
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (src0->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F32) {
            ಆದರೆ (ggml_is_contiguous(src1)) {
                ಹಿಂತಿರುಗಿ ctx->device->pipeline_conv2d_dw_whcn_f16_f32;
            } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (ggml_is_contiguous_channels(src1)) {
                ಹಿಂತಿರುಗಿ ctx->device->pipeline_conv2d_dw_cwhn_f16_f32;
            }
        }
        ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
    ಸ್ಥಿತಿ GGML_OP_CONV_3D:
        ಆದರೆ (src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            ಸ್ಥಿರಾಂಕ uint32_t OC = (uint32_t)ggml_get_op_params_i32(dst, 11);
            ಸ್ಥಿರಾಂಕ uint32_t IC = (uint32_t)ggml_get_op_params_i32(dst, 9);
            ಸ್ಥಿರಾಂಕ uint32_t N  = (uint32_t)ggml_get_op_params_i32(dst, 10);
            ಸ್ಥಿರಾಂಕ uint32_t NPQ = N * dst->ne[2] * dst->ne[1] * dst->ne[0];
            ಸ್ಥಿರಾಂಕ vk_conv_shapes shape = ggml_vk_conv_select_shape(ctx, OC, NPQ);

            ಸ್ಥಿರಾಂಕ uint32_t KW = (uint32_t)src0->ne[0];
            ಸ್ಥಿರಾಂಕ uint32_t KH = (uint32_t)src0->ne[1];
            ಸ್ಥಿರಾಂಕ uint32_t KD = (uint32_t)src0->ne[2];
            ಸ್ಥಿರಾಂಕ uint32_t s0 = (uint32_t)ggml_get_op_params_i32(dst, 0);
            ಸ್ಥಿರಾಂಕ uint32_t s1 = (uint32_t)ggml_get_op_params_i32(dst, 1);
            ಸ್ಥಿರಾಂಕ uint32_t s2 = (uint32_t)ggml_get_op_params_i32(dst, 2);
            ಸ್ಥಿರಾಂಕ uint32_t p0 = (uint32_t)ggml_get_op_params_i32(dst, 3);
            ಸ್ಥಿರಾಂಕ uint32_t p1 = (uint32_t)ggml_get_op_params_i32(dst, 4);
            ಸ್ಥಿರಾಂಕ uint32_t p2 = (uint32_t)ggml_get_op_params_i32(dst, 5);
            ಸ್ಥಿರಾಂಕ uint32_t d0 = (uint32_t)ggml_get_op_params_i32(dst, 6);
            ಸ್ಥಿರಾಂಕ uint32_t d1 = (uint32_t)ggml_get_op_params_i32(dst, 7);
            ಸ್ಥಿರಾಂಕ uint32_t d2 = (uint32_t)ggml_get_op_params_i32(dst, 8);

            ಸ್ಥಿರಾಂಕ uint32_t CRS = IC * KW * KH * KD;
            ಸ್ಥಿರಾಂಕ uint32_t BS_K   = vk_conv_ಖಂಡ_sizes[shape].K;
            ಸ್ಥಿರಾಂಕ uint32_t BS_CRS = vk_conv_ಖಂಡ_sizes[shape].CRS;
            ಸ್ಥಿರಾಂಕ uint32_t BS_NPQ = vk_conv_ಖಂಡ_sizes[shape].NPQ;
            ಸ್ಥಿರಾಂಕ uint32_t aligned = ((OC  % BS_K   == 0) &&
                                      (CRS % BS_CRS == 0) &&
                                      (NPQ % BS_NPQ == 0)) ? 1u : 0u;

            vk_conv3d_pipeline_state conv3d_pipeline_state(s0, s1, s2, p0, p1, p2, d0, d1, d2, KW, KH, KD, aligned);

            std::map<vk_conv3d_pipeline_state, vk_pipeline> *pipelines = ಶೂನ್ಯಸೂಚಕ;
            ಆದರೆ (src0->type == GGML_TYPE_F32) {
                pipelines = &ctx->device->pipeline_conv3d_f32[shape];
            } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (src0->type == GGML_TYPE_F16) {
                pipelines = &ctx->device->pipeline_conv3d_f16_f32[shape];
            } ಇಲ್ಲದಿದ್ದರೆ {
                ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
            }

            vk_pipeline pipeline = ಶೂನ್ಯಸೂಚಕ;

            {
                std::lock_guard<std::mutex> guard(ctx->device->compile_mutex);
                ಸ್ವಯಂಚಾಲಿತ it = pipelines->find(conv3d_pipeline_state);
                ಆದರೆ (it != pipelines->end()) {
                    pipeline = it->second;
                } ಇಲ್ಲದಿದ್ದರೆ {
                    (*pipelines)[conv3d_pipeline_state] = pipeline = std::make_shared<vk_pipeline_struct>();
                }
            }

            ಹಿಂತಿರುಗಿ pipeline;
        }
        ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
    ಸ್ಥಿತಿ GGML_OP_ADD1:
        ಆದರೆ (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F16) {
            ಹಿಂತಿರುಗಿ ctx->device->pipeline_add1_f16_f16;
        }
        ಆದರೆ (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F16) {
            ಹಿಂತಿರುಗಿ ctx->device->pipeline_add1_f16_f32;
        }
        ಆದರೆ (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            ಹಿಂತಿರುಗಿ ctx->device->pipeline_add1_f32_f32;
        }
        ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
    ಸ್ಥಿತಿ GGML_OP_ARANGE:
        ಆದರೆ (dst->type == GGML_TYPE_F32) {
            ಹಿಂತಿರುಗಿ ctx->device->pipeline_arange_f32;
        }
        ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
    ಸ್ಥಿತಿ GGML_OP_FILL:
        ಆದರೆ (dst->type == GGML_TYPE_F32) {
            ಹಿಂತಿರುಗಿ ctx->device->pipeline_fill_f32;
        }
        ಆದರೆ (dst->type == GGML_TYPE_F16) {
            ಹಿಂತಿರುಗಿ ctx->device->pipeline_fill_f16;
        }
        ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
    ಪೂರ್ವನಿಯೋಜಿತ:
        ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
    }

    GGML_ಬಳಸದ(src2);
}

ಮಾದರಿ<ಬಗೆ_ಹೆಸರು PC>
ಸ್ಥಿರ ಶೂನ್ಯ ggml_vk_op_f32(ggml_backend_vk_context * ctx, vk_context& subctx, ಸ್ಥಿರಾಂಕ ggml_tensor * src0, ಸ್ಥಿರಾಂಕ ggml_tensor * src1, ಸ್ಥಿರಾಂಕ ggml_tensor * src2, ಸ್ಥಿರಾಂಕ ggml_tensor * src3, ggml_tensor * dst, ggml_op op, PC&& pc, vk_pipeline pipeline_override = ಶೂನ್ಯಸೂಚಕ) {
    VK_LOG_DEBUG("ggml_vk_op_f32((" << src0 << ", name=" << src0->name << ", type=" << src0->type << ", ne0=" << src0->ne[0] << ", ne1=" << src0->ne[1] << ", ne2=" << src0->ne[2] << ", ne3=" << src0->ne[3] << ", nb0=" << src0->nb[0] << ", nb1=" << src0->nb[1] << ", nb2=" << src0->nb[2] << ", nb3=" << src0->nb[3];
    ಆದರೆ (src1 != ಶೂನ್ಯಸೂಚಕ) {
        std::cerr << "), (" << src1 << ", name=" << src1->name << ", type=" << src1->type << ", ne0=" << src1->ne[0] << ", ne1=" << src1->ne[1] << ", ne2=" << src1->ne[2] << ", ne3=" << src1->ne[3] << ", nb0=" << src1->nb[0] << ", nb1=" << src1->nb[1] << ", nb2=" << src1->nb[2] << ", nb3=" << src1->nb[3];
    }
    ಆದರೆ (src2 != ಶೂನ್ಯಸೂಚಕ) {
        std::cerr << "), (" << src2 << ", name=" << src2->name << ", type=" << src2->type << ", ne0=" << src2->ne[0] << ", ne1=" << src2->ne[1] << ", ne2=" << src2->ne[2] << ", ne3=" << src2->ne[3] << ", nb0=" << src2->nb[0] << ", nb1=" << src2->nb[1] << ", nb2=" << src2->nb[2] << ", nb3=" << src2->nb[3];
    }
    ಆದರೆ (src3 != ಶೂನ್ಯಸೂಚಕ) {
        std::cerr << "), (" << src3 << ", name=" << src3->name << ", type=" << src3->type << ", ne0=" << src3->ne[0] << ", ne1=" << src3->ne[1] << ", ne2=" << src3->ne[2] << ", ne3=" << src3->ne[3] << ", nb0=" << src3->nb[0] << ", nb1=" << src3->nb[1] << ", nb2=" << src3->nb[2] << ", nb3=" << src3->nb[3];
    }
    std::cerr << "), (" << dst << ", name=" << dst->name << ", type=" << dst->type << ", ne0=" << dst->ne[0] << ", ne1=" << dst->ne[1] << ", ne2=" << dst->ne[2] << ", ne3=" << dst->ne[3] << ", nb0=" << dst->nb[0] << ", nb1=" << dst->nb[1] << ", nb2=" << dst->nb[2] << ", nb3=" << dst->nb[3];
    std::cerr << "), " << ggml_op_name(op) << ")");
    GGML_ASSERT(op == GGML_OP_GET_ಅಡ್ಡಸಾಲುಗಳು || op == GGML_OP_CPY || op == GGML_OP_CONCAT || (!ggml_is_quantized(src0->type) && (src1 == ಶೂನ್ಯಸೂಚಕ || !ggml_is_quantized(src1->type))));  // NOLINT
    GGML_ASSERT(dst->ಬಫರ್ != ಶೂನ್ಯಸೂಚಕ);
    ಸ್ಥಿರಾಂಕ uint64_t ne00 = src0->ne[0];
    ಸ್ಥಿರಾಂಕ uint64_t ne01 = src0->ne[1];
    ಸ್ಥಿರಾಂಕ uint64_t ne02 = src0->ne[2];
    ಸ್ಥಿರಾಂಕ uint64_t ne03 = src0->ne[3];

    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ use_src1 = src1 != ಶೂನ್ಯಸೂಚಕ;
    ಸ್ಥಿರಾಂಕ uint64_t ne10 = use_src1 ? src1->ne[0] : 0;
    ಸ್ಥಿರಾಂಕ uint64_t ne11 = use_src1 ? src1->ne[1] : 0;
    ಸ್ಥಿರಾಂಕ uint64_t ne12 = use_src1 ? src1->ne[2] : 0;
    ಸ್ಥಿರಾಂಕ uint64_t ne13 = use_src1 ? src1->ne[3] : 0;

    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ use_src2 = src2 != ಶೂನ್ಯಸೂಚಕ;
    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ use_src3 = src3 != ಶೂನ್ಯಸೂಚಕ;

    init_pushconst_fastdiv(pc);

    vk_pipeline pipeline;
    ಆದರೆ (pipeline_override) {
        pipeline = pipeline_override;
    } ಇಲ್ಲದಿದ್ದರೆ {
        pipeline = ggml_vk_op_get_pipeline(ctx, src0, src1, src2, dst, op);
    }

    ಆದರೆ (pipeline == ಶೂನ್ಯಸೂಚಕ) {
        std::cerr << "ggml_vulkan: Error: Missing op: " << ggml_op_name(op) << " for " << ggml_type_name(src0->type);
        ಆದರೆ (src1 != ಶೂನ್ಯಸೂಚಕ) {
            std::cerr << " and " << ggml_type_name(src1->type);
        }
        std::cerr << " to " << ggml_type_name(dst->type) << std::ಸಾಲು_ಅಂತ್ಯ;
        GGML_ABORT("fatal error");
    }

    ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);

    vk_subbuffer src0_buf = ggml_vk_tensor_subbuffer(ctx, src0, ನಿಜ);
    vk_subbuffer src1_buf = use_src1 ? ggml_vk_tensor_subbuffer(ctx, src1, ನಿಜ) : vk_subbuffer{};
    vk_subbuffer src2_buf = use_src2 ? ggml_vk_tensor_subbuffer(ctx, src2, ನಿಜ) : vk_subbuffer{};
    vk_subbuffer src3_buf = use_src3 ? ggml_vk_tensor_subbuffer(ctx, src3, ನಿಜ) : vk_subbuffer{};
    vk_subbuffer dst_buf = ggml_vk_tensor_subbuffer(ctx, dst, ನಿಜ);

    // Compute misalignment offset for descriptors and store it in in push constants.
    init_pushconst_tensor_offsets(ctx, pc, src0, src1, src2, src3, dst);

    std::array<uint32_t, 3> elements;

    ಆಯ್ಕೆ (op) {
    ಸ್ಥಿತಿ GGML_OP_NORM:
    ಸ್ಥಿತಿ GGML_OP_RMS_NORM_ಹಿಂಭಾಗ:
    ಸ್ಥಿತಿ GGML_OP_L2_NORM:
    ಸ್ಥಿತಿ GGML_OP_SOFT_MAX:
    ಸ್ಥಿತಿ GGML_OP_SOFT_MAX_ಹಿಂಭಾಗ:
    ಸ್ಥಿತಿ GGML_OP_ಮೊತ್ತ_ಅಡ್ಡಸಾಲುಗಳು:
    ಸ್ಥಿತಿ GGML_OP_CUMSUM:
    ಸ್ಥಿತಿ GGML_OP_ಸರಾಸರಿ:
    ಸ್ಥಿತಿ GGML_OP_ARGMAX:
        {
            ಸ್ಥಿರಾಂಕ uint32_t nr = ggml_nrows(src0);
            ಆದರೆ (nr > 262144) {
                elements = { 512, 512, CEIL_DIV(nr, 262144) };
            } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (nr > 512) {
                elements = { 512, CEIL_DIV(nr, 512), 1 };
            } ಇಲ್ಲದಿದ್ದರೆ {
                elements = { nr, 1, 1 };
            }
        } ನಿಲ್ಲಿಸು;
    ಸ್ಥಿತಿ GGML_OP_SOLVE_TRI:
        {
            uint32_t nr = (uint32_t)(ne02 * ne03);
            ಆದರೆ (nr > 262144) {
                elements = { 512, 512, CEIL_DIV(nr, 262144) };
            } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (nr > 512) {
                elements = { 512, CEIL_DIV(nr, 512), 1 };
            } ಇಲ್ಲದಿದ್ದರೆ {
                elements = { nr, 1, 1 };
            }
        }
        ನಿಲ್ಲಿಸು;
    ಸ್ಥಿತಿ GGML_OP_RMS_NORM:
        ಆದರೆ (ctx->do_ಕೂಡಿಸು_rms_partials) {
            // Run one element per thread, 128 threads per workgroup
            elements = { (uint32_t)CEIL_DIV(ne00, 128), 1, 1 };
        } ಇಲ್ಲದಿದ್ದರೆ {
            elements = { (uint32_t)ne01, (uint32_t)ne02, (uint32_t)ne03 };
            elements[1] = std::min(elements[1], ctx->device->properties.limits.max_Compute_Work_ಗುಂಪು_ಎಣಿಕೆ[1]);
            elements[2] = std::min(elements[2], ctx->device->properties.limits.max_Compute_Work_ಗುಂಪು_ಎಣಿಕೆ[2]);
        }
        ನಿಲ್ಲಿಸು;

    ಸ್ಥಿತಿ GGML_OP_ಮೊತ್ತ:
        // We use GGML_OP_SUM_ROWS with 1 row.
        elements = { 1, 1, 1 };
        ನಿಲ್ಲಿಸು;
    ಸ್ಥಿತಿ GGML_OP_ಗುಂಪು_NORM:
        {
            ಸ್ಥಿರಾಂಕ uint32_t ಅಂಕೆ_groups = dst->op_params[0];
            elements = { ಅಂಕೆ_groups * (uint32_t)src0->ne[3], 1, 1 };
        } ನಿಲ್ಲಿಸು;
    ಸ್ಥಿತಿ GGML_OP_DIAG_ಮುಖವಾಡ_INF:
        elements = { (uint32_t)ggml_nrows(src0), (uint32_t)ne00, 1 };
        ನಿಲ್ಲಿಸು;
    ಸ್ಥಿತಿ GGML_OP_ROPE:
    ಸ್ಥಿತಿ GGML_OP_ROPE_ಹಿಂಭಾಗ:
        {
            uint32_t nrows = (uint32_t)ggml_nrows(src0);
            uint32_t z = 1;
            ಆದರೆ (nrows > ctx->device->properties.limits.max_Compute_Work_ಗುಂಪು_ಎಣಿಕೆ[0]) {
                z = CEIL_DIV(nrows, 32768);
                nrows = 32768;
            }
            elements = { nrows, (uint32_t)ne00, z };

        } ನಿಲ್ಲಿಸು;
    ಸ್ಥಿತಿ GGML_OP_GET_ಅಡ್ಡಸಾಲುಗಳು:
        elements = { (uint32_t)ne00, (uint32_t)ne10, (uint32_t)(ne11 * ne12) };
        elements[1] = std::min(elements[1], ctx->device->properties.limits.max_Compute_Work_ಗುಂಪು_ಎಣಿಕೆ[1]);
        elements[2] = std::min(elements[2], ctx->device->properties.limits.max_Compute_Work_ಗುಂಪು_ಎಣಿಕೆ[2]);
        ನಿಲ್ಲಿಸು;
    ಸ್ಥಿತಿ GGML_OP_GET_ಅಡ್ಡಸಾಲುಗಳು_ಹಿಂಭಾಗ:
        elements = { (uint32_t)dst->ne[0], (uint32_t)dst->ne[1], 1 };
        elements[1] = std::min(elements[1], ctx->device->properties.limits.max_Compute_Work_ಗುಂಪು_ಎಣಿಕೆ[1]);
        ನಿಲ್ಲಿಸು;
    ಸ್ಥಿತಿ GGML_OP_ARGSORT:
        GGML_ASSERT(0);
        ನಿಲ್ಲಿಸು;
    ಸ್ಥಿತಿ GGML_OP_IM2_ಲಂಬಸಾಲು:
        {
            ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ is_2D = dst->op_params[6] == 1;

            ಸ್ಥಿರಾಂಕ uint32_t IC = src1->ne[is_2D ? 2 : 1];

            ಸ್ಥಿರಾಂಕ uint32_t KH = is_2D ? src0->ne[1] : 1;
            ಸ್ಥಿರಾಂಕ uint32_t KW =         src0->ne[0];

            ಸ್ಥಿರಾಂಕ uint32_t OH = is_2D ? dst->ne[2] : 1;
            ಸ್ಥಿರಾಂಕ uint32_t OW =         dst->ne[1];

            ಸ್ಥಿರಾಂಕ uint32_t batch = src1->ne[is_2D ? 3 : 2];

            ಸ್ಥಿರಾಂಕ uint32_t CHW = IC * KH * KW;
            // Cap X workgroups to limit concurrent IC channel reads.
            // The shader loops over X to cover the full CHW dimension.
            // AMD prefers a lower limit
            ಸ್ಥಿರಾಂಕ uint32_t min_cap = ctx->device->vendor_id == VK_VENDOR_ID_AMD ? 512u : 4096u;
            ಸ್ಥಿರಾಂಕ uint32_t x_elements = std::min(CHW, std::max(min_cap, OW * KH * KW));
            elements = { x_elements, OW, OH * batch };
            elements[1] = std::min(elements[1], ctx->device->properties.limits.max_Compute_Work_ಗುಂಪು_ಎಣಿಕೆ[1]);
            elements[2] = std::min(elements[2], ctx->device->properties.limits.max_Compute_Work_ಗುಂಪು_ಎಣಿಕೆ[2]);
        } ನಿಲ್ಲಿಸು;
    ಸ್ಥಿತಿ GGML_OP_IM2_ಲಂಬಸಾಲು_3_D:
        {
            ಸ್ಥಿರಾಂಕ uint32_t IC = ((ಸ್ಥಿರಾಂಕ uint32_t *)(dst->op_params))[9];

            ಸ್ಥಿರಾಂಕ uint32_t N  = ne13 / IC;

            ಸ್ಥಿರಾಂಕ uint32_t KD = ne02;
            ಸ್ಥಿರಾಂಕ uint32_t KH = ne01;
            ಸ್ಥಿರಾಂಕ uint32_t KW = ne00;

            ಸ್ಥಿರಾಂಕ uint32_t OD = dst->ne[3] / N;
            ಸ್ಥಿರಾಂಕ uint32_t OH = dst->ne[2];
            ಸ್ಥಿರಾಂಕ uint32_t OW = dst->ne[1];

            ಸ್ಥಿರಾಂಕ uint32_t IC_KD_KH_KW = IC*KD*KH*KW;
            ಸ್ಥಿರಾಂಕ uint32_t N_OD_OH = N*OD*OH;

            elements = { IC_KD_KH_KW, OW, N_OD_OH };
            elements[2] = std::min(elements[2], ctx->device->properties.limits.max_Compute_Work_ಗುಂಪು_ಎಣಿಕೆ[2]);
        } ನಿಲ್ಲಿಸು;
    ಸ್ಥಿತಿ GGML_OP_TIMESTEP_EMBEDDING:
        {
            ಸ್ಥಿರಾಂಕ uint32_t dim = dst->op_params[0];
            uint32_t half_ceil = (dim + 1) / 2;
            elements = { half_ceil, (uint32_t)src0->ne[0], 1 };
        } ನಿಲ್ಲಿಸು;
    ಸ್ಥಿತಿ GGML_OP_CONV_TRANSPOSE_1D:
        {
            elements = {uint32_t(src0->ne[1]), 1, 1}; // parallelize in {Cout, 1, 1}
        } ನಿಲ್ಲಿಸು;
    ಸ್ಥಿತಿ GGML_OP_COL2IM_1D:
        {
            elements = { uint32_t(dst->ne[0]), uint32_t(dst->ne[1]), 1 };
        } ನಿಲ್ಲಿಸು;
    ಸ್ಥಿತಿ GGML_OP_POOL_1D:
        {
            ಸ್ಥಿರಾಂಕ uint32_t N = dst->ne[3] * dst->ne[2];
            ಸ್ಥಿರಾಂಕ uint32_t OC = dst->ne[1];
            ಸ್ಥಿರಾಂಕ uint32_t OL = dst->ne[0];
            elements = { N * OC * OL, 1, 1};
        } ನಿಲ್ಲಿಸು;
    ಸ್ಥಿತಿ GGML_OP_POOL_2D:
        {
            ಸ್ಥಿರಾಂಕ uint32_t outputs = ggml_nelements(dst);
            ಸ್ಥಿರಾಂಕ uint32_t outputs_ಪ್ರತಿ_workgroup = pipeline == ctx->device->pipeline_pool2d_subgroup_f32 ? 128 / ctx->device->subgroup_ಗಾತ್ರ : 512;
            ಸ್ಥಿರಾಂಕ uint32_t workgroups = CEIL_DIV(outputs, outputs_ಪ್ರತಿ_workgroup);
            ಸ್ಥಿರಾಂಕ uint32_t groups_y = CEIL_DIV(workgroups, ctx->device->properties.limits.max_Compute_Work_ಗುಂಪು_ಎಣಿಕೆ[0]);
            ಸ್ಥಿರಾಂಕ uint32_t groups_x = CEIL_DIV(workgroups, groups_y);
            elements = { groups_x, groups_y, 1 };
        } ನಿಲ್ಲಿಸು;
    ಸ್ಥಿತಿ GGML_OP_CONV_2D:
    ಸ್ಥಿತಿ GGML_OP_CONV_TRANSPOSE_2D:
        ಆದರೆ constexpr (std::is_same_v<PC, vk_op_conv2d_ತಳ್ಳು_constants>) {
            ಸ್ಥಿರಾಂಕ uint32_t NPQ = pc.N * pc.OH * pc.OW;
            ಸ್ಥಿರಾಂಕ vk_conv_shapes shape = ggml_vk_conv_select_shape(ctx, pc.Cout, NPQ);
            ಸ್ಥಿರಾಂಕ uint32_t NPQ_ಖಂಡಗಳು = CEIL_DIV(NPQ, vk_conv_ಖಂಡ_sizes[shape].NPQ);

            elements = { pc.Cout, NPQ_ಖಂಡಗಳು, 1 };
            ಆದರೆ (elements[1] > 512) {
                elements[2] = CEIL_DIV(elements[1], 512);
                elements[1] = 512;
            }
        } ಇಲ್ಲದಿದ್ದರೆ {
            GGML_ABORT("invalid push constant type for CONV_2D");
        }
        ನಿಲ್ಲಿಸು;
    ಸ್ಥಿತಿ GGML_OP_CONV_3D:
        ಆದರೆ constexpr (std::is_same_v<PC, vk_op_conv3d_ತಳ್ಳು_constants>) {
            ಸ್ಥಿರಾಂಕ uint32_t NPQ = pc.N * pc.OD * pc.OH * pc.OW;
            ಸ್ಥಿರಾಂಕ vk_conv_shapes shape = ggml_vk_conv_select_shape(ctx, pc.OC, NPQ);
            ಸ್ಥಿರಾಂಕ uint32_t NPQ_ಖಂಡಗಳು = CEIL_DIV(NPQ, vk_conv_ಖಂಡ_sizes[shape].NPQ);

            elements = { pc.OC, NPQ_ಖಂಡಗಳು, 1 };
            ಆದರೆ (elements[1] > 512) {
                elements[2] = CEIL_DIV(elements[1], 512);
                elements[1] = 512;
            }
        } ಇಲ್ಲದಿದ್ದರೆ {
            GGML_ABORT("invalid push constant type for CONV_3D");
        }
        ನಿಲ್ಲಿಸು;
    ಸ್ಥಿತಿ GGML_OP_ಕೂಡಿಸು:
    ಸ್ಥಿತಿ GGML_OP_SUB:
    ಸ್ಥಿತಿ GGML_OP_DIV:
    ಸ್ಥಿತಿ GGML_OP_MUL:
    ಸ್ಥಿತಿ GGML_OP_ADD1:
    ಸ್ಥಿತಿ GGML_OP_ಹೊರಹೋಗು_PROD:
    ಸ್ಥಿತಿ GGML_OP_ARANGE:
    ಸ್ಥಿತಿ GGML_OP_FILL:
    ಸ್ಥಿತಿ GGML_OP_SCALE:
    ಸ್ಥಿತಿ GGML_OP_SQR:
    ಸ್ಥಿತಿ GGML_OP_SQRT:
    ಸ್ಥಿತಿ GGML_OP_SIN:
    ಸ್ಥಿತಿ GGML_OP_COS:
    ಸ್ಥಿತಿ GGML_OP_LOG:
    ಸ್ಥಿತಿ GGML_OP_TRI:
    ಸ್ಥಿತಿ GGML_OP_DIAG:
    ಸ್ಥಿತಿ GGML_OP_CLAMP:
    ಸ್ಥಿತಿ GGML_OP_LEAKY_RELU:
    ಸ್ಥಿತಿ GGML_OP_PAD:
    ಸ್ಥಿತಿ GGML_OP_PAD_REFLECT_1D:
    ಸ್ಥಿತಿ GGML_OP_ROLL:
    ಸ್ಥಿತಿ GGML_OP_REPEAT:
    ಸ್ಥಿತಿ GGML_OP_REPEAT_ಹಿಂಭಾಗ:
    ಸ್ಥಿತಿ GGML_OP_CPY:
    ಸ್ಥಿತಿ GGML_OP_CONCAT:
    ಸ್ಥಿತಿ GGML_OP_UPSCALE:
    ಸ್ಥಿತಿ GGML_OP_UNARY:
    ಸ್ಥಿತಿ GGML_OP_GLU:
    ಸ್ಥಿತಿ GGML_OP_CONV_2D_DW:
        {
            uint32_t ne = ggml_nelements(dst);
            ಆದರೆ (op == GGML_OP_CPY && ggml_is_quantized(src0->type) && ggml_is_quantized(dst->type)) {
                // Convert from number of logical elements to 2- or 4-byte units.
                ne /= ggml_blck_ಗಾತ್ರ(src0->type);
                ಆದರೆ ((ggml_type_ಗಾತ್ರ(src0->type) % 4) == 0) {
                    ne *= ggml_type_ಗಾತ್ರ(src0->type) / 4;
                } ಇಲ್ಲದಿದ್ದರೆ {
                    ne *= ggml_type_ಗಾತ್ರ(src0->type) / 2;
                }
            }
            ಆದರೆ (op == GGML_OP_CONCAT && ggml_is_quantized(dst->type)) {
                ne = ne / ggml_blck_ಗಾತ್ರ(dst->type) * ggml_type_ಗಾತ್ರ(dst->type) / ggml_vk_concat_unit_ಗಾತ್ರ(dst->type);
            }
            // copy_to_quant has block size of 32, and each thread does QUANT_K elements.
            // Splitting into 512x512xZ wouldn't work well since each workgroup does 1024 elements.
            // So divide by block size here before splitting into 512x512 groups.
            ಆದರೆ (op == GGML_OP_CPY && !ggml_is_quantized(src0->type) && ggml_is_quantized(dst->type)) {
                ne = CEIL_DIV(ne, ggml_blck_ಗಾತ್ರ(dst->type));
            }
            ಆದರೆ (ne > 262144) {
                elements = { 512, 512, CEIL_DIV(ne, 262144) };
            } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (ne > 512) {
                elements = { 512, CEIL_DIV(ne, 512), 1 };
            } ಇಲ್ಲದಿದ್ದರೆ {
                elements = { ne, 1, 1 };
            }

            ಆದರೆ (pipeline == ctx->device->pipeline_cpy_transpose_02_32 ||
                pipeline == ctx->device->pipeline_cpy_transpose_02_16) {
                // 32x32 tiles over dims 0 and 2; dim1 and dim3 are the batch
                elements[0] = (uint32_t)CEIL_DIV(dst->ne[0], 32);
                elements[1] = (uint32_t)CEIL_DIV(dst->ne[2], 32);
                elements[2] = (uint32_t)(dst->ne[1]*dst->ne[3]);
                elements[0] = std::min(elements[0], ctx->device->properties.limits.max_Compute_Work_ಗುಂಪು_ಎಣಿಕೆ[0]);
                elements[1] = std::min(elements[1], ctx->device->properties.limits.max_Compute_Work_ಗುಂಪು_ಎಣಿಕೆ[1]);
                elements[2] = std::min(elements[2], ctx->device->properties.limits.max_Compute_Work_ಗುಂಪು_ಎಣಿಕೆ[2]);
            } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (pipeline == ctx->device->pipeline_cpy_transpose_32 ||
                pipeline == ctx->device->pipeline_cpy_transpose_16) {
                // 32x32 tiles
                elements[0] = (uint32_t)CEIL_DIV(dst->ne[0], 32);
                elements[1] = (uint32_t)CEIL_DIV(dst->ne[1], 32);
                elements[2] = (uint32_t)(dst->ne[2]*dst->ne[3]);
                elements[0] = std::min(elements[0], ctx->device->properties.limits.max_Compute_Work_ಗುಂಪು_ಎಣಿಕೆ[0]);
                elements[1] = std::min(elements[1], ctx->device->properties.limits.max_Compute_Work_ಗುಂಪು_ಎಣಿಕೆ[1]);
                elements[2] = std::min(elements[2], ctx->device->properties.limits.max_Compute_Work_ಗುಂಪು_ಎಣಿಕೆ[2]);
            }
        } ನಿಲ್ಲಿಸು;
    ಸ್ಥಿತಿ GGML_OP_ಕೂಡಿಸು_ID:
        {
            elements = { (uint32_t)ne01, (uint32_t)ne02, 1 };
        } ನಿಲ್ಲಿಸು;
    ಸ್ಥಿತಿ GGML_OP_ಗಣ_ಅಡ್ಡಸಾಲುಗಳು:
        {
            uint32_t ne = ggml_nelements(src0);
            ಆದರೆ (ggml_is_quantized(dst->type)) {
                // quants run 32 threads each doing QUANT_K elements
                ne = CEIL_DIV(ne, 32 * ggml_blck_ಗಾತ್ರ(dst->type));
            } ಇಲ್ಲದಿದ್ದರೆ {
                // scalar types do one element per thread, running 512 threads
                ne = CEIL_DIV(ne, 512);
            }
            ಆದರೆ (ne > 262144) {
                elements = { 512, 512, CEIL_DIV(ne, 262144) };
            } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (ne > 512) {
                elements = { 512, CEIL_DIV(ne, 512), 1 };
            } ಇಲ್ಲದಿದ್ದರೆ {
                elements = { ne, 1, 1 };
            }
        }
        ನಿಲ್ಲಿಸು;
    ಸ್ಥಿತಿ GGML_OP_SSM_CONV:
        {
            ಸ್ಥಿರಾಂಕ uint32_t nr  = src0->ne[1];
            ಸ್ಥಿರಾಂಕ uint32_t n_t = dst->ne[1];
            ಸ್ಥಿರಾಂಕ uint32_t n_s = dst->ne[2];
            elements = { nr, n_t, n_s };
        }
        ನಿಲ್ಲಿಸು;
    ಪೂರ್ವನಿಯೋಜಿತ:
        elements = { (uint32_t)ggml_nelements(src0), 1, 1 };
        ನಿಲ್ಲಿಸು;
    }

    ಆದರೆ (op == GGML_OP_ಕೂಡಿಸು || op == GGML_OP_RMS_NORM) {
        vk_subbuffer a_buf = src0_buf;
        ಆದರೆ (ctx->do_ಕೂಡಿಸು_rms_partials) {
            a_buf = ggml_vk_subbuffer(ctx, ctx->prealloc_ಕೂಡಿಸು_rms_partials, ctx->prealloc_ಗಾತ್ರ_ಕೂಡಿಸು_rms_partials_ಸ್ಥಾನಾಂತರ);
        }
        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline,
            { src0_buf, src1_buf, dst_buf, a_buf }, pc, elements);
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (op == GGML_OP_GLU) {
        // Empty src1 is possible in glu, but the shader needs a buffer
        vk_subbuffer subbuf1 = use_src1 ? src1_buf : src0_buf;
        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { src0_buf, subbuf1, dst_buf }, pc, elements);
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (op == GGML_OP_SOFT_MAX) {
        // Empty src1 and src2 is possible in soft_max, but the shader needs a buffer
        vk_subbuffer subbuf1 = use_src1 ? src1_buf : src0_buf;
        vk_subbuffer subbuf2 = use_src2 ? src2_buf : src0_buf;
        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { src0_buf, subbuf1, subbuf2, dst_buf }, pc, elements);
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (op == GGML_OP_ROPE || op == GGML_OP_ROPE_ಹಿಂಭಾಗ) {
        // Empty src2 and src3 is possible in rope, but the shader needs a buffer
        vk_subbuffer subbuf2 = use_src2 ? src2_buf : src0_buf;
        vk_subbuffer subbuf3 = use_src3 ? src3_buf : src0_buf;
        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { src0_buf, src1_buf, subbuf2, dst_buf, subbuf3 }, pc, elements);
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (op == GGML_OP_IM2_ಲಂಬಸಾಲು || op == GGML_OP_IM2_ಲಂಬಸಾಲು_3_D) {
        ಆದರೆ (ctx->device->shader_int64 && ctx->device->ಬಫರ್_device_ವಿಳಾಸ) {
            // buffer device address path doesn't use dst buffer
            dst_buf.size = 1;
        }
        // im2col uses only src1 and dst buffers
        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { src1_buf, dst_buf }, pc, elements);
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (op == GGML_OP_ಎಣಿಕೆ_EQUAL) {
        // count_equal assumes that destination buffer is initialized with zeroes
        ggml_vk_ಬಫರ್_memset_async(subctx, dst_buf.ಬಫರ್, dst_buf.ಸ್ಥಾನಾಂತರ, 0, dst_buf.size);
        ggml_vk_sync_buffers(ctx, subctx);
        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { src0_buf, src1_buf, dst_buf }, pc, elements);
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (op == GGML_OP_OPT_ಹೆಜ್ಜೆ_SGD) {
        // OPT_STEP_SGD works on src0, it does not need dst
        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { src0_buf, src1_buf, src2_buf }, pc, elements);
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (use_src3) {
        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { src0_buf, src1_buf, src2_buf, src3_buf, dst_buf }, pc, elements);
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (use_src2) {
        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { src0_buf, src1_buf, src2_buf, dst_buf }, pc, elements);
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (use_src1) {
        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { src0_buf, src1_buf, dst_buf }, pc, elements);
    } ಇಲ್ಲದಿದ್ದರೆ {
        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { src0_buf, dst_buf }, pc, elements);
    }
}

ಶೂನ್ಯ ggml_vk_get_ಅಡ್ಡಸಾಲುಗಳು(ggml_backend_vk_context * ctx, vk_context& subctx, ಸ್ಥಿರಾಂಕ ggml_tensor * src0, ಸ್ಥಿರಾಂಕ ggml_tensor * src1, ggml_tensor * dst) {
    ಸ್ಥಿರಾಂಕ uint32_t src0_type_ಗಾತ್ರ = ggml_type_ಗಾತ್ರ(src0->type);
    ಸ್ಥಿರಾಂಕ uint32_t src1_type_ಗಾತ್ರ = ggml_type_ಗಾತ್ರ(src1->type);
    ಸ್ಥಿರಾಂಕ uint32_t dst_type_ಗಾತ್ರ = ggml_type_ಗಾತ್ರ(dst->type);

    ggml_vk_op_f32<vk_op_ದ್ವಿಮಾನ_ತಳ್ಳು_constants>(ctx, subctx, src0, src1, ಶೂನ್ಯಸೂಚಕ, ಶೂನ್ಯಸೂಚಕ, dst, GGML_OP_GET_ಅಡ್ಡಸಾಲುಗಳು, {
        (uint32_t)ggml_nelements(src0),
        (uint32_t)src0->ne[0], (uint32_t)src0->ne[1], (uint32_t)src0->ne[2],(uint32_t)src0->ne[3], (uint32_t)src0->nb[0] / src0_type_ಗಾತ್ರ, (uint32_t)src0->nb[1] / src0_type_ಗಾತ್ರ, (uint32_t)src0->nb[2] / src0_type_ಗಾತ್ರ, (uint32_t)src0->nb[3] / src0_type_ಗಾತ್ರ,
        (uint32_t)src1->ne[0], (uint32_t)src1->ne[1], (uint32_t)src1->ne[2],(uint32_t)src1->ne[3], (uint32_t)src1->nb[0] / src1_type_ಗಾತ್ರ, (uint32_t)src1->nb[1] / src1_type_ಗಾತ್ರ, (uint32_t)src1->nb[2] / src1_type_ಗಾತ್ರ, (uint32_t)src1->nb[3] / src1_type_ಗಾತ್ರ,
        (uint32_t) dst->ne[0], (uint32_t) dst->ne[1], (uint32_t) dst->ne[2],(uint32_t) dst->ne[3], (uint32_t) dst->nb[0] /  dst_type_ಗಾತ್ರ, (uint32_t) dst->nb[1] /  dst_type_ಗಾತ್ರ, (uint32_t) dst->nb[2] /  dst_type_ಗಾತ್ರ, (uint32_t) dst->nb[3] /  dst_type_ಗಾತ್ರ,
        0,
        0.0f, 0.0f, 0,
    });
}

ಶೂನ್ಯ ggml_vk_get_ಅಡ್ಡಸಾಲುಗಳು_ಹಿಂಭಾಗ(ggml_backend_vk_context * ctx, vk_context& subctx, ಸ್ಥಿರಾಂಕ ggml_tensor * src0, ಸ್ಥಿರಾಂಕ ggml_tensor * src1, ggml_tensor * dst) {
    ಸ್ಥಿರಾಂಕ uint32_t src0_type_ಗಾತ್ರ = ggml_type_ಗಾತ್ರ(src0->type);
    ಸ್ಥಿರಾಂಕ uint32_t src1_type_ಗಾತ್ರ = ggml_type_ಗಾತ್ರ(src1->type);
    ಸ್ಥಿರಾಂಕ uint32_t dst_type_ಗಾತ್ರ = ggml_type_ಗಾತ್ರ(dst->type);

    ggml_vk_op_f32<vk_op_ದ್ವಿಮಾನ_ತಳ್ಳು_constants>(ctx, subctx, src0, src1, ಶೂನ್ಯಸೂಚಕ, ಶೂನ್ಯಸೂಚಕ, dst, GGML_OP_GET_ಅಡ್ಡಸಾಲುಗಳು_ಹಿಂಭಾಗ, {
        (uint32_t)ggml_nelements(src0),
        (uint32_t)src0->ne[0], (uint32_t)src0->ne[1], (uint32_t)src0->ne[2], (uint32_t)src0->ne[3], (uint32_t)src0->nb[0] / src0_type_ಗಾತ್ರ, (uint32_t)src0->nb[1] / src0_type_ಗಾತ್ರ, (uint32_t)src0->nb[2] / src0_type_ಗಾತ್ರ, (uint32_t)src0->nb[3] / src0_type_ಗಾತ್ರ,
        (uint32_t)src1->ne[0], (uint32_t)src1->ne[1], (uint32_t)src1->ne[2], (uint32_t)src1->ne[3], (uint32_t)src1->nb[0] / src1_type_ಗಾತ್ರ, (uint32_t)src1->nb[1] / src1_type_ಗಾತ್ರ, (uint32_t)src1->nb[2] / src1_type_ಗಾತ್ರ, (uint32_t)src1->nb[3] / src1_type_ಗಾತ್ರ,
        (uint32_t) dst->ne[0], (uint32_t) dst->ne[1], (uint32_t) dst->ne[2], (uint32_t) dst->ne[3], (uint32_t) dst->nb[0] /  dst_type_ಗಾತ್ರ, (uint32_t) dst->nb[1] /  dst_type_ಗಾತ್ರ, (uint32_t) dst->nb[2] /  dst_type_ಗಾತ್ರ, (uint32_t) dst->nb[3] /  dst_type_ಗಾತ್ರ,
        0,
        0.0f, 0.0f, 0,
    });
}

ಶೂನ್ಯ ggml_vk_acc(ggml_backend_vk_context * ctx, vk_context& subctx, ಸ್ಥಿರಾಂಕ ggml_tensor * src0, ಸ್ಥಿರಾಂಕ ggml_tensor * src1, ggml_tensor * dst) {
    ಸ್ಥಿರಾಂಕ uint32_t src0_type_ಗಾತ್ರ = ggml_type_ಗಾತ್ರ(src0->type);
    ಸ್ಥಿರಾಂಕ uint32_t src1_type_ಗಾತ್ರ = ggml_type_ಗಾತ್ರ(src1->type);
    ಸ್ಥಿರಾಂಕ uint32_t dst_type_ಗಾತ್ರ = ggml_type_ಗಾತ್ರ(dst->type);

    ಪೂರ್ಣಾಂಕ nb1 = dst->op_params[0] / src0_type_ಗಾತ್ರ; // 4 bytes of float32
    ಪೂರ್ಣಾಂಕ nb2 = dst->op_params[1] / src0_type_ಗಾತ್ರ; // 4 bytes of float32
    ಪೂರ್ಣಾಂಕ nb3 = dst->op_params[2] / src0_type_ಗಾತ್ರ; // 4 bytes of float32
    ಪೂರ್ಣಾಂಕ ಸ್ಥಾನಾಂತರ = dst->op_params[3] / src0_type_ಗಾತ್ರ; // offset in bytes

    ggml_vk_op_f32<vk_op_ದ್ವಿಮಾನ_ತಳ್ಳು_constants>(ctx, subctx, src0, src1, ಶೂನ್ಯಸೂಚಕ, ಶೂನ್ಯಸೂಚಕ, dst, dst->op, {
        (uint32_t)ggml_nelements(src0),
        (uint32_t)src0->ne[0], (uint32_t)src0->ne[1], (uint32_t)src0->ne[2],(uint32_t)src0->ne[3], (uint32_t)src0->nb[0] / src0_type_ಗಾತ್ರ, (uint32_t)nb1, (uint32_t)nb2, (uint32_t)nb3,
        (uint32_t)src1->ne[0], (uint32_t)src1->ne[1], (uint32_t)src1->ne[2],(uint32_t)src1->ne[3], (uint32_t)src1->nb[0] / src1_type_ಗಾತ್ರ, (uint32_t)src1->nb[1] / src1_type_ಗಾತ್ರ, (uint32_t)src1->nb[2] / src1_type_ಗಾತ್ರ, (uint32_t)src1->nb[3] / src1_type_ಗಾತ್ರ,
        (uint32_t) dst->ne[0], (uint32_t) dst->ne[1], (uint32_t) dst->ne[2],(uint32_t) dst->ne[3], (uint32_t) dst->nb[0] /  dst_type_ಗಾತ್ರ, (uint32_t)nb1, (uint32_t)nb2, (uint32_t)nb3,
        0,
        0.0f, 0.0f, ಸ್ಥಾನಾಂತರ,
    });
}

ಶೂನ್ಯ ggml_vk_multi_ಕೂಡಿಸು(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_cgraph * cgraph, ಪೂರ್ಣಾಂಕ ಗಂಟು_idx) {
    ಸ್ಥಿರಾಂಕ ggml_tensor *ಮೊದಲ_ಗಂಟು = cgraph->nodes[ಗಂಟು_idx];
    ಸ್ಥಿರಾಂಕ ggml_tensor *dst = cgraph->nodes[ಗಂಟು_idx + ctx->ಅಂಕೆ_additional_fused_ops];

    // Make a list of all the tensors used by the op.
    // Last element of the list is the dest tensor.
    ಸ್ಥಿರಾಂಕ ggml_tensor *tensors[MAX_PARAMETER_ಎಣಿಕೆ];
    uint32_t ಅಂಕೆ_srcs = ctx->ಅಂಕೆ_additional_fused_ops + 2;
    uint32_t ಅಂಕೆ_tensors = ಅಂಕೆ_srcs + 1;
    GGML_ASSERT(ಅಂಕೆ_tensors + ctx->do_ಕೂಡಿಸು_rms_partials <= MAX_PARAMETER_ಎಣಿಕೆ);

    tensors[0] = ಮೊದಲ_ಗಂಟು->src[0];
    tensors[1] = ಮೊದಲ_ಗಂಟು->src[1];
    ಗಾಗಿ (int32_t i = 0; i < ctx->ಅಂಕೆ_additional_fused_ops; ++i) {
        // check whether the previous result is src[0] or src[1]
        ಆದರೆ (cgraph->nodes[ಗಂಟು_idx + i] == cgraph->nodes[ಗಂಟು_idx + i + 1]->src[0]) {
            tensors[i+2] = cgraph->nodes[ಗಂಟು_idx + i + 1]->src[1];
        } ಇಲ್ಲದಿದ್ದರೆ {
            tensors[i+2] = cgraph->nodes[ಗಂಟು_idx + i + 1]->src[0];
        }
    }
    tensors[ಅಂಕೆ_srcs] = dst;

    vk_op_multi_ಕೂಡಿಸು_ತಳ್ಳು_constants pc;
    pc.ne20 = (uint32_t)dst->ne[0];
    pc.ne21 = (uint32_t)dst->ne[1];
    pc.ne22 = (uint32_t)dst->ne[2];
    pc.ne23 = (uint32_t)dst->ne[3];

    ಗಾಗಿ (uint32_t i = 0; i < ಅಂಕೆ_tensors; ++i) {
        ಸ್ಥಿರಾಂಕ ggml_tensor *t = tensors[i];
        pc.nb[i][0] = (uint32_t)t->nb[0] / ಗಾತ್ರ(ದಶಮಾಂಶ);
        pc.nb[i][1] = (uint32_t)t->nb[1] / ಗಾತ್ರ(ದಶಮಾಂಶ);
        pc.nb[i][2] = (uint32_t)t->nb[2] / ಗಾತ್ರ(ದಶಮಾಂಶ);
        pc.nb[i][3] = (uint32_t)t->nb[3] / ಗಾತ್ರ(ದಶಮಾಂಶ);
    }
    pc.rms_partials = ctx->do_ಕೂಡಿಸು_rms_partials;

    vk_pipeline pipeline = ggml_vk_op_get_pipeline(ctx, tensors[0], tensors[1], ಶೂನ್ಯಸೂಚಕ, dst, dst->op);

    ಆದರೆ (pipeline == ಶೂನ್ಯಸೂಚಕ) {
        std::cerr << "ggml_vulkan: Error: Missing multi_add";
        GGML_ABORT("fatal error");
    }

    ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);

    ggml_backend_vk_ಬಫರ್_context * buf_ctx[MAX_PARAMETER_ಎಣಿಕೆ];
    vk_ಬಫರ್ buf[MAX_PARAMETER_ಎಣಿಕೆ];
    size_t ಸ್ಥಾನಾಂತರ[MAX_PARAMETER_ಎಣಿಕೆ];
    ತಾರ್ಕಿಕ uma[MAX_PARAMETER_ಎಣಿಕೆ];

    ಗಾಗಿ (uint32_t i = 0; i < ಅಂಕೆ_tensors; ++i) {
        buf_ctx[i] = (ggml_backend_vk_ಬಫರ್_context *)tensors[i]->ಬಫರ್->context;
        buf[i] = ಶೂನ್ಯಸೂಚಕ;
        ಸ್ಥಾನಾಂತರ[i] = 0;
        uma[i] = ಸುಳ್ಳು;

        ಆದರೆ (ctx->device->uma) {
            ggml_vk_host_get(ctx->device, tensors[i]->ದತ್ತಾಂಶ, buf[i], ಸ್ಥಾನಾಂತರ[i]);
            uma[i] = buf[i] != ಶೂನ್ಯಸೂಚಕ;
        }
        ಆದರೆ (!uma[i]) {
            buf[i] = buf_ctx[i]->dev_ಬಫರ್;
            ಸ್ಥಾನಾಂತರ[i] = vk_tensor_ಸ್ಥಾನಾಂತರ(tensors[i]) + tensors[i]->view_offs;
        }
        GGML_ASSERT(buf[i] != ಶೂನ್ಯಸೂಚಕ);
    }
    // If any remaining descriptors are unused, just point them at src[0]
    ಗಾಗಿ (uint32_t i = ಅಂಕೆ_tensors; i < MAX_PARAMETER_ಎಣಿಕೆ; ++i) {
        buf[i] = buf[0];
        ಸ್ಥಾನಾಂತರ[i] = 0;
    }
    ಆದರೆ (ctx->do_ಕೂಡಿಸು_rms_partials) {
        buf[ಅಂಕೆ_tensors] = ctx->prealloc_ಕೂಡಿಸು_rms_partials;
        ಸ್ಥಾನಾಂತರ[ಅಂಕೆ_tensors] = ctx->prealloc_ಗಾತ್ರ_ಕೂಡಿಸು_rms_partials_ಸ್ಥಾನಾಂತರ;
    }

    std::array<uint32_t, 3> elements;

    uint32_t ne = ggml_nelements(dst);
    ಆದರೆ (ne > 262144) {
        elements = { 512, 512, CEIL_DIV(ne, 262144) };
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (ne > 512) {
        elements = { 512, CEIL_DIV(ne, 512), 1 };
    } ಇಲ್ಲದಿದ್ದರೆ {
        elements = { ne, 1, 1 };
    }

    static_assert(MAX_PARAMETER_ಎಣಿಕೆ == 12);
    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline,
        {
            ggml_vk_subbuffer(ctx, buf[0], ಸ್ಥಾನಾಂತರ[0]),
            ggml_vk_subbuffer(ctx, buf[1], ಸ್ಥಾನಾಂತರ[1]),
            ggml_vk_subbuffer(ctx, buf[2], ಸ್ಥಾನಾಂತರ[2]),
            ggml_vk_subbuffer(ctx, buf[3], ಸ್ಥಾನಾಂತರ[3]),
            ggml_vk_subbuffer(ctx, buf[4], ಸ್ಥಾನಾಂತರ[4]),
            ggml_vk_subbuffer(ctx, buf[5], ಸ್ಥಾನಾಂತರ[5]),
            ggml_vk_subbuffer(ctx, buf[6], ಸ್ಥಾನಾಂತರ[6]),
            ggml_vk_subbuffer(ctx, buf[7], ಸ್ಥಾನಾಂತರ[7]),
            ggml_vk_subbuffer(ctx, buf[8], ಸ್ಥಾನಾಂತರ[8]),
            ggml_vk_subbuffer(ctx, buf[9], ಸ್ಥಾನಾಂತರ[9]),
            ggml_vk_subbuffer(ctx, buf[10], ಸ್ಥಾನಾಂತರ[10]),
            ggml_vk_subbuffer(ctx, buf[11], ಸ್ಥಾನಾಂತರ[11]),
        }, pc, elements);
}

ಶೂನ್ಯ ggml_vk_ಕೂಡಿಸು(ggml_backend_vk_context * ctx, vk_context& subctx, ಸ್ಥಿರಾಂಕ ggml_tensor * src0, ಸ್ಥಿರಾಂಕ ggml_tensor * src1, ggml_tensor * dst) {
    ಸ್ಥಿರಾಂಕ uint32_t src0_type_ಗಾತ್ರ = ggml_type_ಗಾತ್ರ(src0->type);
    ಸ್ಥಿರಾಂಕ uint32_t src1_type_ಗಾತ್ರ = ggml_type_ಗಾತ್ರ(src1->type);
    ಸ್ಥಿರಾಂಕ uint32_t dst_type_ಗಾತ್ರ = ggml_type_ಗಾತ್ರ(dst->type);

    ggml_vk_op_f32<vk_op_ದ್ವಿಮಾನ_ತಳ್ಳು_constants>(ctx, subctx, src0, src1, ಶೂನ್ಯಸೂಚಕ, ಶೂನ್ಯಸೂಚಕ, dst, GGML_OP_ಕೂಡಿಸು, {
        (uint32_t)ggml_nelements(src0),
        (uint32_t)src0->ne[0], (uint32_t)src0->ne[1], (uint32_t)src0->ne[2],(uint32_t)src0->ne[3], (uint32_t)src0->nb[0] / src0_type_ಗಾತ್ರ, (uint32_t)src0->nb[1] / src0_type_ಗಾತ್ರ, (uint32_t)src0->nb[2] / src0_type_ಗಾತ್ರ, (uint32_t)src0->nb[3] / src0_type_ಗಾತ್ರ,
        (uint32_t)src1->ne[0], (uint32_t)src1->ne[1], (uint32_t)src1->ne[2],(uint32_t)src1->ne[3], (uint32_t)src1->nb[0] / src1_type_ಗಾತ್ರ, (uint32_t)src1->nb[1] / src1_type_ಗಾತ್ರ, (uint32_t)src1->nb[2] / src1_type_ಗಾತ್ರ, (uint32_t)src1->nb[3] / src1_type_ಗಾತ್ರ,
        (uint32_t) dst->ne[0], (uint32_t) dst->ne[1], (uint32_t) dst->ne[2],(uint32_t) dst->ne[3], (uint32_t) dst->nb[0] /  dst_type_ಗಾತ್ರ, (uint32_t) dst->nb[1] /  dst_type_ಗಾತ್ರ, (uint32_t) dst->nb[2] /  dst_type_ಗಾತ್ರ, (uint32_t) dst->nb[3] /  dst_type_ಗಾತ್ರ,
        0,
        0.0f, 0.0f, ctx->do_ಕೂಡಿಸು_rms_partials,
    });
}

ಶೂನ್ಯ ggml_vk_ಹೊರಹೋಗು_prod(ggml_backend_vk_context * ctx, vk_context& subctx, ಸ್ಥಿರಾಂಕ ggml_tensor * src0, ಸ್ಥಿರಾಂಕ ggml_tensor * src1, ggml_tensor * dst) {
    ಸ್ಥಿರಾಂಕ uint32_t src0_type_ಗಾತ್ರ = ggml_type_ಗಾತ್ರ(src0->type);
    ಸ್ಥಿರಾಂಕ uint32_t src1_type_ಗಾತ್ರ = ggml_type_ಗಾತ್ರ(src1->type);
    ಸ್ಥಿರಾಂಕ uint32_t dst_type_ಗಾತ್ರ = ggml_type_ಗಾತ್ರ(dst->type);

    ggml_vk_op_f32<vk_op_ದ್ವಿಮಾನ_ತಳ್ಳು_constants>(ctx, subctx, src0, src1, ಶೂನ್ಯಸೂಚಕ, ಶೂನ್ಯಸೂಚಕ, dst, GGML_OP_ಹೊರಹೋಗು_PROD, {
        (uint32_t)ggml_nelements(dst),
        (uint32_t)src0->ne[0], (uint32_t)src0->ne[1], (uint32_t)src0->ne[2],(uint32_t)src0->ne[3],
        (uint32_t)src0->nb[0] / src0_type_ಗಾತ್ರ, (uint32_t)src0->nb[1] / src0_type_ಗಾತ್ರ, (uint32_t)src0->nb[2] / src0_type_ಗಾತ್ರ, (uint32_t)src0->nb[3] / src0_type_ಗಾತ್ರ,
        (uint32_t)src1->ne[0], (uint32_t)src1->ne[1], (uint32_t)src1->ne[2],(uint32_t)src1->ne[3],
        (uint32_t)src1->nb[0] / src1_type_ಗಾತ್ರ, (uint32_t)src1->nb[1] / src1_type_ಗಾತ್ರ, (uint32_t)src1->nb[2] / src1_type_ಗಾತ್ರ, (uint32_t)src1->nb[3] / src1_type_ಗಾತ್ರ,
        (uint32_t) dst->ne[0], (uint32_t) dst->ne[1], (uint32_t) dst->ne[2],(uint32_t) dst->ne[3],
        (uint32_t) dst->nb[0] /  dst_type_ಗಾತ್ರ, (uint32_t) dst->nb[1] /  dst_type_ಗಾತ್ರ, (uint32_t) dst->nb[2] /  dst_type_ಗಾತ್ರ, (uint32_t) dst->nb[3] /  dst_type_ಗಾತ್ರ,
        0,
        0.0f, 0.0f, 0,
    });
}

ಶೂನ್ಯ ggml_vk_sub(ggml_backend_vk_context * ctx, vk_context& subctx, ಸ್ಥಿರಾಂಕ ggml_tensor * src0, ಸ್ಥಿರಾಂಕ ggml_tensor * src1, ggml_tensor * dst) {
    ಸ್ಥಿರಾಂಕ uint32_t src0_type_ಗಾತ್ರ = ggml_type_ಗಾತ್ರ(src0->type);
    ಸ್ಥಿರಾಂಕ uint32_t src1_type_ಗಾತ್ರ = ggml_type_ಗಾತ್ರ(src1->type);
    ಸ್ಥಿರಾಂಕ uint32_t dst_type_ಗಾತ್ರ = ggml_type_ಗಾತ್ರ(dst->type);

    ggml_vk_op_f32<vk_op_ದ್ವಿಮಾನ_ತಳ್ಳು_constants>(ctx, subctx, src0, src1, ಶೂನ್ಯಸೂಚಕ, ಶೂನ್ಯಸೂಚಕ, dst, GGML_OP_SUB, {
        (uint32_t)ggml_nelements(src0),
        (uint32_t)src0->ne[0], (uint32_t)src0->ne[1], (uint32_t)src0->ne[2],(uint32_t)src0->ne[3], (uint32_t)src0->nb[0] / src0_type_ಗಾತ್ರ, (uint32_t)src0->nb[1] / src0_type_ಗಾತ್ರ, (uint32_t)src0->nb[2] / src0_type_ಗಾತ್ರ, (uint32_t)src0->nb[3] / src0_type_ಗಾತ್ರ,
        (uint32_t)src1->ne[0], (uint32_t)src1->ne[1], (uint32_t)src1->ne[2],(uint32_t)src1->ne[3], (uint32_t)src1->nb[0] / src1_type_ಗಾತ್ರ, (uint32_t)src1->nb[1] / src1_type_ಗಾತ್ರ, (uint32_t)src1->nb[2] / src1_type_ಗಾತ್ರ, (uint32_t)src1->nb[3] / src1_type_ಗಾತ್ರ,
        (uint32_t) dst->ne[0], (uint32_t) dst->ne[1], (uint32_t) dst->ne[2],(uint32_t) dst->ne[3], (uint32_t) dst->nb[0] /  dst_type_ಗಾತ್ರ, (uint32_t) dst->nb[1] /  dst_type_ಗಾತ್ರ, (uint32_t) dst->nb[2] /  dst_type_ಗಾತ್ರ, (uint32_t) dst->nb[3] /  dst_type_ಗಾತ್ರ,
        0,
        0.0f, 0.0f, 0,
    });
}

ಶೂನ್ಯ ggml_vk_mul(ggml_backend_vk_context * ctx, vk_context& subctx, ಸ್ಥಿರಾಂಕ ggml_tensor * src0, ಸ್ಥಿರಾಂಕ ggml_tensor * src1, ggml_tensor * dst) {
    ಸ್ಥಿರಾಂಕ uint32_t src0_type_ಗಾತ್ರ = ggml_type_ಗಾತ್ರ(src0->type);
    ಸ್ಥಿರಾಂಕ uint32_t src1_type_ಗಾತ್ರ = ggml_type_ಗಾತ್ರ(src1->type);
    ಸ್ಥಿರಾಂಕ uint32_t dst_type_ಗಾತ್ರ = ggml_type_ಗಾತ್ರ(dst->type);

    ggml_vk_op_f32<vk_op_ದ್ವಿಮಾನ_ತಳ್ಳು_constants>(ctx, subctx, src0, src1, ಶೂನ್ಯಸೂಚಕ, ಶೂನ್ಯಸೂಚಕ, dst, GGML_OP_MUL, {
        (uint32_t)ggml_nelements(src0),
        (uint32_t)src0->ne[0], (uint32_t)src0->ne[1], (uint32_t)src0->ne[2],(uint32_t)src0->ne[3], (uint32_t)src0->nb[0] / src0_type_ಗಾತ್ರ, (uint32_t)src0->nb[1] / src0_type_ಗಾತ್ರ, (uint32_t)src0->nb[2] / src0_type_ಗಾತ್ರ, (uint32_t)src0->nb[3] / src0_type_ಗಾತ್ರ,
        (uint32_t)src1->ne[0], (uint32_t)src1->ne[1], (uint32_t)src1->ne[2],(uint32_t)src1->ne[3], (uint32_t)src1->nb[0] / src1_type_ಗಾತ್ರ, (uint32_t)src1->nb[1] / src1_type_ಗಾತ್ರ, (uint32_t)src1->nb[2] / src1_type_ಗಾತ್ರ, (uint32_t)src1->nb[3] / src1_type_ಗಾತ್ರ,
        (uint32_t) dst->ne[0], (uint32_t) dst->ne[1], (uint32_t) dst->ne[2],(uint32_t) dst->ne[3], (uint32_t) dst->nb[0] /  dst_type_ಗಾತ್ರ, (uint32_t) dst->nb[1] /  dst_type_ಗಾತ್ರ, (uint32_t) dst->nb[2] /  dst_type_ಗಾತ್ರ, (uint32_t) dst->nb[3] /  dst_type_ಗಾತ್ರ,
        0,
        0.0f, 0.0f, 0,
    });
}

ಪೂರ್ಣಾಂಕ ggml_vk_unary_mul_op_ಸೂಚ್ಯಂಕ(ggml_unary_op op) {
    ಆಯ್ಕೆ (op) {
        ಸ್ಥಿತಿ GGML_UNARY_OP_GELU:     ಹಿಂತಿರುಗಿ 0;
        ಸ್ಥಿತಿ GGML_UNARY_OP_SIGMOID:  ಹಿಂತಿರುಗಿ 1;
        ಸ್ಥಿತಿ GGML_UNARY_OP_SILU:     ಹಿಂತಿರುಗಿ 2;
        ಸ್ಥಿತಿ GGML_UNARY_OP_SOFTPLUS: ಹಿಂತಿರುಗಿ 3;
        ಪೂರ್ವನಿಯೋಜಿತ:                     ಹಿಂತಿರುಗಿ -1;
    }
}

ಶೂನ್ಯ ggml_vk_unary_mul(ggml_backend_vk_context * ctx, vk_context& subctx, ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_cgraph * cgraph, ಪೂರ್ಣಾಂಕ ಗಂಟು_idx) {
    ಸ್ಥಿರಾಂಕ ggml_tensor * unary = cgraph->nodes[ಗಂಟು_idx];
    ggml_tensor * mul = cgraph->nodes[ಗಂಟು_idx + 1];

    // unary on src1 that tiles into src0
    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ op_on_b = mul->src[1] == unary &&
                         !ggml_are_same_shape(unary->src[0], mul->src[0]) &&
                         ggml_can_repeat(unary, mul->src[0]);

    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = op_on_b ? mul->src[0] : unary->src[0];
    ಸ್ಥಿರಾಂಕ ggml_tensor * src1 = op_on_b ? unary->src[0] :
        ((mul->src[0] == unary) ? mul->src[1] : mul->src[0]);

    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ f16 = src0->type == GGML_TYPE_F16;
    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ norepeat = ggml_are_same_shape(src0, src1);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ oi = ggml_vk_unary_mul_op_ಸೂಚ್ಯಂಕ(ggml_get_unary_op(unary));
    ಆದರೆ (oi < 0) {
        GGML_ABORT("fatal error");
    }
    vk_pipeline pipeline = ctx->device->pipeline_unary_mul[oi][f16][norepeat][op_on_b];

    ಸ್ಥಿರಾಂಕ uint32_t src0_type_ಗಾತ್ರ = ggml_type_ಗಾತ್ರ(src0->type);
    ಸ್ಥಿರಾಂಕ uint32_t src1_type_ಗಾತ್ರ = ggml_type_ಗಾತ್ರ(src1->type);
    ಸ್ಥಿರಾಂಕ uint32_t dst_type_ಗಾತ್ರ = ggml_type_ಗಾತ್ರ(mul->type);

    ggml_vk_op_f32<vk_op_ದ್ವಿಮಾನ_ತಳ್ಳು_constants>(ctx, subctx, src0, src1, ಶೂನ್ಯಸೂಚಕ, ಶೂನ್ಯಸೂಚಕ, mul, GGML_OP_UNARY, {
        (uint32_t)ggml_nelements(op_on_b ? mul : src0),
        (uint32_t)src0->ne[0], (uint32_t)src0->ne[1], (uint32_t)src0->ne[2],(uint32_t)src0->ne[3], (uint32_t)src0->nb[0] / src0_type_ಗಾತ್ರ, (uint32_t)src0->nb[1] / src0_type_ಗಾತ್ರ, (uint32_t)src0->nb[2] / src0_type_ಗಾತ್ರ, (uint32_t)src0->nb[3] / src0_type_ಗಾತ್ರ,
        (uint32_t)src1->ne[0], (uint32_t)src1->ne[1], (uint32_t)src1->ne[2],(uint32_t)src1->ne[3], (uint32_t)src1->nb[0] / src1_type_ಗಾತ್ರ, (uint32_t)src1->nb[1] / src1_type_ಗಾತ್ರ, (uint32_t)src1->nb[2] / src1_type_ಗಾತ್ರ, (uint32_t)src1->nb[3] / src1_type_ಗಾತ್ರ,
        (uint32_t) mul->ne[0], (uint32_t) mul->ne[1], (uint32_t) mul->ne[2],(uint32_t) mul->ne[3], (uint32_t) mul->nb[0] /  dst_type_ಗಾತ್ರ, (uint32_t) mul->nb[1] /  dst_type_ಗಾತ್ರ, (uint32_t) mul->nb[2] /  dst_type_ಗಾತ್ರ, (uint32_t) mul->nb[3] /  dst_type_ಗಾತ್ರ,
        0,
        0.0f, 0.0f, 0,
    }, pipeline);
}

ಶೂನ್ಯ ggml_vk_div(ggml_backend_vk_context * ctx, vk_context& subctx, ಸ್ಥಿರಾಂಕ ggml_tensor * src0, ಸ್ಥಿರಾಂಕ ggml_tensor * src1, ggml_tensor * dst) {
    ಸ್ಥಿರಾಂಕ uint32_t src0_type_ಗಾತ್ರ = ggml_type_ಗಾತ್ರ(src0->type);
    ಸ್ಥಿರಾಂಕ uint32_t src1_type_ಗಾತ್ರ = ggml_type_ಗಾತ್ರ(src1->type);
    ಸ್ಥಿರಾಂಕ uint32_t dst_type_ಗಾತ್ರ = ggml_type_ಗಾತ್ರ(dst->type);

    ggml_vk_op_f32<vk_op_ದ್ವಿಮಾನ_ತಳ್ಳು_constants>(ctx, subctx, src0, src1, ಶೂನ್ಯಸೂಚಕ, ಶೂನ್ಯಸೂಚಕ, dst, GGML_OP_DIV, {
        (uint32_t)ggml_nelements(src0),
        (uint32_t)src0->ne[0], (uint32_t)src0->ne[1], (uint32_t)src0->ne[2],(uint32_t)src0->ne[3], (uint32_t)src0->nb[0] / src0_type_ಗಾತ್ರ, (uint32_t)src0->nb[1] / src0_type_ಗಾತ್ರ, (uint32_t)src0->nb[2] / src0_type_ಗಾತ್ರ, (uint32_t)src0->nb[3] / src0_type_ಗಾತ್ರ,
        (uint32_t)src1->ne[0], (uint32_t)src1->ne[1], (uint32_t)src1->ne[2],(uint32_t)src1->ne[3], (uint32_t)src1->nb[0] / src1_type_ಗಾತ್ರ, (uint32_t)src1->nb[1] / src1_type_ಗಾತ್ರ, (uint32_t)src1->nb[2] / src1_type_ಗಾತ್ರ, (uint32_t)src1->nb[3] / src1_type_ಗಾತ್ರ,
        (uint32_t) dst->ne[0], (uint32_t) dst->ne[1], (uint32_t) dst->ne[2],(uint32_t) dst->ne[3], (uint32_t) dst->nb[0] /  dst_type_ಗಾತ್ರ, (uint32_t) dst->nb[1] /  dst_type_ಗಾತ್ರ, (uint32_t) dst->nb[2] /  dst_type_ಗಾತ್ರ, (uint32_t) dst->nb[3] /  dst_type_ಗಾತ್ರ,
        0,
        0.0f, 0.0f, 0,
    });
}

ಶೂನ್ಯ ggml_vk_ಕೂಡಿಸು_id(ggml_backend_vk_context * ctx, vk_context& subctx, ಸ್ಥಿರಾಂಕ ggml_tensor * src0, ಸ್ಥಿರಾಂಕ ggml_tensor * src1, ಸ್ಥಿರಾಂಕ ggml_tensor * src2, ggml_tensor * dst) {
    ಸ್ಥಿರಾಂಕ uint32_t src0_type_ಗಾತ್ರ = ggml_type_ಗಾತ್ರ(src0->type);
    ಸ್ಥಿರಾಂಕ uint32_t src1_type_ಗಾತ್ರ = ggml_type_ಗಾತ್ರ(src1->type);
    ಸ್ಥಿರಾಂಕ uint32_t src2_type_ಗಾತ್ರ = ggml_type_ಗಾತ್ರ(src2->type);

    ggml_vk_op_f32<vk_op_ಕೂಡಿಸು_id_ತಳ್ಳು_constants>(ctx, subctx, src0, src1, src2, ಶೂನ್ಯಸೂಚಕ, dst, GGML_OP_ಕೂಡಿಸು_ID, {
        (uint32_t)dst->ne[0],
        (uint32_t)dst->ne[1],
        (uint32_t)src0->nb[1] / src0_type_ಗಾತ್ರ,
        (uint32_t)src0->nb[2] / src0_type_ಗಾತ್ರ,
        (uint32_t)src1->nb[1] / src1_type_ಗಾತ್ರ,
        (uint32_t)src2->nb[1] / src2_type_ಗಾತ್ರ,
    });
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_vk_op_f32_wkv(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst, ಸ್ಥಿರಾಂಕ vk_op_rwkv_wkv6_ತಳ್ಳು_constants&& pc, ಪೂರ್ಣಾಂಕ ಆವೃತ್ತಿ_2) {
    GGML_ASSERT(ಆವೃತ್ತಿ_2 == 6 || ಆವೃತ್ತಿ_2 == 7);
    ಪೂರ್ಣಾಂಕ ಅಂಕೆ_srcs = ಆವೃತ್ತಿ_2 == 6 ? 6 : 7;

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < ಅಂಕೆ_srcs; i++) {
        GGML_ASSERT(!ggml_is_quantized(dst->src[i]->type));
    }

    GGML_ASSERT(dst->ಬಫರ್ != ಶೂನ್ಯಸೂಚಕ);

    vk_pipeline pipeline = ggml_vk_op_get_pipeline(ctx, dst->src[0], dst->src[1], dst->src[2], dst, dst->op);
    GGML_ASSERT(pipeline != ಶೂನ್ಯಸೂಚಕ);

    ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);

    vk_subbuffer dst_buf = ggml_vk_tensor_subbuffer(ctx, dst);
    vk_subbuffer src_buf[7] = {};
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < ಅಂಕೆ_srcs; i++) {
        src_buf[i] = ggml_vk_tensor_subbuffer(ctx, dst->src[i]);
    }

    std::array<uint32_t, 3> elements = {
        (uint32_t)(pc.B * pc.H),
        1,
        1
    };

    ಆದರೆ (ಆವೃತ್ತಿ_2 == 6) {
        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline,
            {src_buf[0], src_buf[1], src_buf[2], src_buf[3], src_buf[4], src_buf[5], dst_buf},
            pc, elements);
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (ಆವೃತ್ತಿ_2 == 7) {
        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline,
            {src_buf[0], src_buf[1], src_buf[2], src_buf[3], src_buf[4], src_buf[5], src_buf[6], dst_buf},
            pc, elements);
    } ಇಲ್ಲದಿದ್ದರೆ {
        // shouldn't happen
        GGML_ASSERT(ಸುಳ್ಳು);
    }
}

ಶೂನ್ಯ ggml_vk_rwkv_wkv6(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst) {
    ಸ್ಥಿರಾಂಕ size_t seq_ಉದ್ದಳತೆ = dst->src[0]->ne[2];
    ಸ್ಥಿರಾಂಕ size_t n_embed = dst->ne[0];
    ಸ್ಥಿರಾಂಕ size_t n_heads = dst->src[0]->ne[1];
    ಸ್ಥಿರಾಂಕ size_t n_seqs = dst->src[5]->ne[1];

    ggml_vk_op_f32_wkv(
        ctx, subctx, dst,
        {
            (uint32_t)n_seqs,
            (uint32_t)seq_ಉದ್ದಳತೆ,
            (uint32_t)n_embed,
            (uint32_t)n_heads,
        },
        6
    );
}

ಶೂನ್ಯ ggml_vk_rwkv_wkv7(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst) {
    ಸ್ಥಿರಾಂಕ size_t seq_ಉದ್ದಳತೆ = dst->src[0]->ne[2];
    ಸ್ಥಿರಾಂಕ size_t n_embed = dst->ne[0];
    ಸ್ಥಿರಾಂಕ size_t n_heads = dst->src[0]->ne[1];
    ಸ್ಥಿರಾಂಕ size_t n_seqs = dst->src[6]->ne[1];

    ggml_vk_op_f32_wkv(
        ctx, subctx, dst,
        {
            (uint32_t)n_seqs,
            (uint32_t)seq_ಉದ್ದಳತೆ,
            (uint32_t)n_embed,
            (uint32_t)n_heads,
        },
        7
    );
}

ಶೂನ್ಯ ggml_vk_gated_ರೇಖೀಯ_attn(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst) {
    ಸ್ಥಿರಾಂಕ size_t seq_ಉದ್ದಳತೆ = dst->src[0]->ne[2];
    ಸ್ಥಿರಾಂಕ size_t n_embed    = dst->ne[0];
    ಸ್ಥಿರಾಂಕ size_t n_heads    = dst->src[0]->ne[1];
    ಸ್ಥಿರಾಂಕ size_t n_seqs     = dst->src[4]->ne[1];

    ದಶಮಾಂಶ scale;
    ಸ್ಮರಣೆ_ನಕಲು(&scale, dst->op_params, ಗಾತ್ರ(ದಶಮಾಂಶ));

    GGML_ASSERT(dst->ಬಫರ್ != ಶೂನ್ಯಸೂಚಕ);

    vk_pipeline pipeline = ggml_vk_op_get_pipeline(ctx, dst->src[0], dst->src[1], dst->src[2], dst, dst->op);
    GGML_ASSERT(pipeline != ಶೂನ್ಯಸೂಚಕ);

    ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);

    vk_subbuffer dst_buf = ggml_vk_tensor_subbuffer(ctx, dst);
    vk_subbuffer src_buf[5] = {};
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < 5; i++) {
        src_buf[i] = ggml_vk_tensor_subbuffer(ctx, dst->src[i]);
    }

    ಸ್ಥಿರಾಂಕ vk_op_gated_ರೇಖೀಯ_attn_ತಳ್ಳು_constants pc = {
        (uint32_t)n_seqs,
        (uint32_t)seq_ಉದ್ದಳತೆ,
        (uint32_t)n_embed,
        (uint32_t)n_heads,
        scale,
    };

    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline,
        {src_buf[0], src_buf[1], src_buf[2], src_buf[3], src_buf[4], dst_buf},
        pc, { (uint32_t)(n_seqs * n_heads), 1, 1 });
}

ಶೂನ್ಯ ggml_vk_lightning_indexer(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst) {
    ಸ್ಥಿರಾಂಕ ggml_tensor * q = dst->src[0];
    ಸ್ಥಿರಾಂಕ ggml_tensor * k = dst->src[1];
    ಸ್ಥಿರಾಂಕ ggml_tensor * w = dst->src[2];
    ಸ್ಥಿರಾಂಕ ggml_tensor * m = dst->src[3];

    vk_pipeline pipeline = ggml_vk_op_get_pipeline(ctx, q, k, w, dst, dst->op);
    GGML_ASSERT(pipeline != ಶೂನ್ಯಸೂಚಕ);

    ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);

    ಸ್ಥಿರಾಂಕ uint32_t n_kv      = k->ne[2];
    ಸ್ಥಿರಾಂಕ uint32_t n_heads   = q->ne[1];
    ಸ್ಥಿರಾಂಕ uint32_t n_ಟೋಕನುಗಳು  = q->ne[2];
    ಸ್ಥಿರಾಂಕ uint32_t n_streams = q->ne[3];
    ಸ್ಥಿರಾಂಕ uint32_t n_masks   = m->ne[3];

    // one workgroup per tile of 64 keys and 8 tokens, see lightning_indexer.comp
    ಸ್ಥಿರಾಂಕ uint32_t n_tiles_kv = CEIL_DIV(n_kv, 64);
    ಸ್ಥಿರಾಂಕ uint32_t n_tiles_t  = CEIL_DIV(n_ಟೋಕನುಗಳು, 8);

    // q, w and dst are f32 and m is f16, so their strides are passed in elements;
    // k may be quantized, so its strides stay in bytes
    ಸ್ಥಿರಾಂಕ uint32_t q_nb1 = q->nb[1] / ಗಾತ್ರ(ದಶಮಾಂಶ);
    ಸ್ಥಿರಾಂಕ uint32_t q_nb2 = q->nb[2] / ಗಾತ್ರ(ದಶಮಾಂಶ);
    ಸ್ಥಿರಾಂಕ uint32_t q_nb3 = q->nb[3] / ಗಾತ್ರ(ದಶಮಾಂಶ);
    ಸ್ಥಿರಾಂಕ uint32_t k_nb2 = k->nb[2];
    ಸ್ಥಿರಾಂಕ uint32_t k_nb3 = k->nb[3];
    ಸ್ಥಿರಾಂಕ uint32_t w_nb1 = w->nb[1] / ಗಾತ್ರ(ದಶಮಾಂಶ);
    ಸ್ಥಿರಾಂಕ uint32_t w_nb3 = w->nb[3] / ಗಾತ್ರ(ದಶಮಾಂಶ);
    ಸ್ಥಿರಾಂಕ uint32_t m_nb1 = m->nb[1] / ಗಾತ್ರ(ggml_fp16_t);
    ಸ್ಥಿರಾಂಕ uint32_t m_nb3 = m->nb[3] / ಗಾತ್ರ(ggml_fp16_t);
    ಸ್ಥಿರಾಂಕ uint32_t d_nb1 = dst->nb[1] / ಗಾತ್ರ(ದಶಮಾಂಶ);
    ಸ್ಥಿರಾಂಕ uint32_t d_nb3 = dst->nb[3] / ಗಾತ್ರ(ದಶಮಾಂಶ);

    ಸ್ಥಿರಾಂಕ vk_op_lightning_indexer_ತಳ್ಳು_constants pc = {
        n_kv, n_heads, n_ಟೋಕನುಗಳು, n_masks,
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
    ಸ್ಥಿರಾಂಕ ggml_tensor * src_q     = dst->src[0];
    ಸ್ಥಿರಾಂಕ ggml_tensor * src_v     = dst->src[2];
    ಸ್ಥಿರಾಂಕ ggml_tensor * src_beta  = dst->src[4];

    GGML_ASSERT(dst->ಬಫರ್ != ಶೂನ್ಯಸೂಚಕ);

    ಸ್ಥಿರಾಂಕ uint32_t S_v      = (uint32_t)src_v->ne[0];
    ಸ್ಥಿರಾಂಕ uint32_t H        = (uint32_t)src_v->ne[1];
    ಸ್ಥಿರಾಂಕ uint32_t n_ಟೋಕನುಗಳು = (uint32_t)src_v->ne[2];
    ಸ್ಥಿರಾಂಕ uint32_t n_seqs   = (uint32_t)src_v->ne[3];

    // K (snapshot slot count) is an op param; state holds s0 only [S_v, S_v, H, n_seqs].
    ಸ್ಥಿರಾಂಕ uint32_t K = (uint32_t)ggml_get_op_params_i32(dst, 0);

    ಸ್ಥಿರಾಂಕ uint32_t s_off = S_v * H * n_ಟೋಕನುಗಳು * n_seqs;

    vk_pipeline pipeline = ggml_vk_op_get_pipeline(ctx, dst->src[0], dst->src[1], dst->src[2], dst, dst->op);
    GGML_ASSERT(pipeline != ಶೂನ್ಯಸೂಚಕ);

    ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);

    vk_subbuffer dst_buf = ggml_vk_tensor_subbuffer(ctx, dst);
    vk_subbuffer src_buf[6] = {};
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < 6; i++) {
        src_buf[i] = ggml_vk_tensor_subbuffer(ctx, dst->src[i]);
    }

    ಸ್ಥಿರಾಂಕ uint32_t sq1 = (uint32_t)(src_q->nb[1] / ಗಾತ್ರ(ದಶಮಾಂಶ));
    ಸ್ಥಿರಾಂಕ uint32_t sq2 = (uint32_t)(src_q->nb[2] / ಗಾತ್ರ(ದಶಮಾಂಶ));
    ಸ್ಥಿರಾಂಕ uint32_t sq3 = (uint32_t)(src_q->nb[3] / ಗಾತ್ರ(ದಶಮಾಂಶ));
    ಸ್ಥಿರಾಂಕ uint32_t sv1 = (uint32_t)(src_v->nb[1] / ಗಾತ್ರ(ದಶಮಾಂಶ));
    ಸ್ಥಿರಾಂಕ uint32_t sv2 = (uint32_t)(src_v->nb[2] / ಗಾತ್ರ(ದಶಮಾಂಶ));
    ಸ್ಥಿರಾಂಕ uint32_t sv3 = (uint32_t)(src_v->nb[3] / ಗಾತ್ರ(ದಶಮಾಂಶ));
    ಸ್ಥಿರಾಂಕ uint32_t sb1 = (uint32_t)(src_beta->nb[1] / ಗಾತ್ರ(ದಶಮಾಂಶ));
    ಸ್ಥಿರಾಂಕ uint32_t sb2 = (uint32_t)(src_beta->nb[2] / ಗಾತ್ರ(ದಶಮಾಂಶ));
    ಸ್ಥಿರಾಂಕ uint32_t sb3 = (uint32_t)(src_beta->nb[3] / ಗಾತ್ರ(ದಶಮಾಂಶ));

    ಸ್ಥಿರಾಂಕ uint32_t neq1 = (uint32_t)src_q->ne[1];
    ಸ್ಥಿರಾಂಕ uint32_t rq3  = (uint32_t)(src_v->ne[3] / src_q->ne[3]);

    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ scale = 1.0f / sqrtf((ದಶಮಾಂಶ)S_v);
    ಸ್ಥಿರಾಂಕ vk_op_gated_delta_net_ತಳ್ಳು_constants pc = {
        H, n_ಟೋಕನುಗಳು, n_seqs, s_off,
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

ಶೂನ್ಯ ggml_vk_ssm_ಪರಿಶೀಲನೆ(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst) {
    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = dst->src[0];
    ಸ್ಥಿರಾಂಕ ggml_tensor * src1 = dst->src[1];
    ಸ್ಥಿರಾಂಕ ggml_tensor * src2 = dst->src[2];
    ಸ್ಥಿರಾಂಕ ggml_tensor * src3 = dst->src[3];
    ಸ್ಥಿರಾಂಕ ggml_tensor * src4 = dst->src[4];
    ಸ್ಥಿರಾಂಕ ggml_tensor * src5 = dst->src[5];

    GGML_ASSERT(dst->ಬಫರ್ != ಶೂನ್ಯಸೂಚಕ);

    ಸ್ಥಿರಾಂಕ uint32_t ತಲೆ_dim = src0->ne[1];
    ಸ್ಥಿರಾಂಕ uint32_t n_ತಲೆ = src1->ne[1];
    ಸ್ಥಿರಾಂಕ uint32_t n_ಗುಂಪು = src4->ne[1];
    ಸ್ಥಿರಾಂಕ uint32_t n_tok = src1->ne[2];
    ಸ್ಥಿರಾಂಕ uint32_t n_seq = src1->ne[3];

    ತಾರ್ಕಿಕ is_mamba2 = (src3->nb[1] == ಗಾತ್ರ(ದಶಮಾಂಶ));
    GGML_ASSERT(is_mamba2);

    vk_pipeline pipeline = ggml_vk_op_get_pipeline(ctx, src0, src1, src2, dst, dst->op);
    GGML_ASSERT(pipeline != ಶೂನ್ಯಸೂಚಕ);

    ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);

    ಸ್ಥಿರಾಂಕ int64_t s_off = ggml_nelements(src1) * ಗಾತ್ರ(ದಶಮಾಂಶ);

    ಸ್ಥಿರಾಂಕ vk_op_ssm_ಪರಿಶೀಲನೆ_ತಳ್ಳು_constants pc = {
        (uint32_t)src0->nb[2], (uint32_t)src0->nb[3],
        (uint32_t)src1->nb[2], (uint32_t)src1->nb[3],
        (uint32_t)src2->nb[1], (uint32_t)src2->nb[2],
        (uint32_t)src3->nb[1],
        (uint32_t)src4->nb[2], (uint32_t)src4->nb[3],
        (uint32_t)src5->nb[2], (uint32_t)src5->nb[3],
        (uint32_t)s_off,
        n_ತಲೆ, ತಲೆ_dim, n_ಗುಂಪು, n_tok,
        n_seq, (uint32_t) ggml_get_op_params_i32(dst, 0)
    };

    vk_subbuffer dst_buf = ggml_vk_tensor_subbuffer(ctx, dst);
    vk_subbuffer src_buf[7] = {};
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < 7 && dst->src[i] != ಶೂನ್ಯಸೂಚಕ; i++) {
        src_buf[i] = ggml_vk_tensor_subbuffer(ctx, dst->src[i]);
    }

    std::array<uint32_t, 3> elements;

    ಸ್ಥಿರಾಂಕ uint32_t d_state = src0->ne[0];
    uint32_t ಅಂಕೆ_subgroups = d_state / ctx->device->subgroup_ಗಾತ್ರ;
    ಸ್ಥಿರಾಂಕ uint32_t ಅಂಕೆ_workgroups_x = CEIL_DIV(n_ತಲೆ * ತಲೆ_dim, ಅಂಕೆ_subgroups);
    ಸ್ಥಿರಾಂಕ uint32_t ಅಂಕೆ_workgroups_y = n_seq;
    elements = { ಅಂಕೆ_workgroups_x, ಅಂಕೆ_workgroups_y, 1 };

    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline,
        {src_buf[0], src_buf[1], src_buf[2], src_buf[3], src_buf[4], src_buf[5], src_buf[6], dst_buf},
        pc, elements);
}

ಶೂನ್ಯ ggml_vk_ssm_conv(ggml_backend_vk_context * ctx, vk_context& subctx, ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_cgraph * cgraph, ಪೂರ್ಣಾಂಕ ಗಂಟು_idx) {
    ggml_tensor * conv = cgraph->nodes[ಗಂಟು_idx];
    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = conv->src[0];
    ಸ್ಥಿರಾಂಕ ggml_tensor * src1 = conv->src[1];

    // Pick the destination tensor (last node in the fused chain) and the optional bias.
    // Fusion modes: 0 = ssm_conv, 1 = ssm_conv+silu, 2 = ssm_conv+add(bias)+silu.
    ggml_tensor * dst = conv;
    ಸ್ಥಿರಾಂಕ ggml_tensor * bias = ಶೂನ್ಯಸೂಚಕ;

    ಆದರೆ (ctx->ಅಂಕೆ_additional_fused_ops == 1) {
        dst = cgraph->nodes[ಗಂಟು_idx + 1]; // silu
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (ctx->ಅಂಕೆ_additional_fused_ops == 2) {
        ggml_tensor * ಕೂಡಿಸು_2 = cgraph->nodes[ಗಂಟು_idx + 1];
        bias = (ಕೂಡಿಸು_2->src[0] == conv) ? ಕೂಡಿಸು_2->src[1] : ಕೂಡಿಸು_2->src[0];
        dst = cgraph->nodes[ಗಂಟು_idx + 2]; // silu
    }

    // The shader always declares 4 bindings; bind src0 as a dummy when bias isn't fused.
    ಸ್ಥಿರಾಂಕ ggml_tensor * src2 = bias ? bias : src0;

    ggml_vk_op_f32<vk_op_ssm_conv_ತಳ್ಳು_constants>(ctx, subctx, src0, src1, src2, ಶೂನ್ಯಸೂಚಕ, dst, GGML_OP_SSM_CONV, {
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

ಸ್ಥಿರ ಶೂನ್ಯ ggml_vk_op_f32_opt_ಹೆಜ್ಜೆ_adamw(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst, ಸ್ಥಿರಾಂಕ vk_op_ತಳ್ಳು_constants&& pc) {
    ಸ್ಥಿರಾಂಕ ggml_tensor * x = dst->src[0];
    ಸ್ಥಿರಾಂಕ ggml_tensor * g = dst->src[1];
    ಸ್ಥಿರಾಂಕ ggml_tensor * gm = dst->src[2];
    ಸ್ಥಿರಾಂಕ ggml_tensor * gv = dst->src[3];
    ಸ್ಥಿರಾಂಕ ggml_tensor * p = dst->src[4];

    GGML_ASSERT(x->type == GGML_TYPE_F32);
    GGML_ASSERT(g->type == GGML_TYPE_F32);
    GGML_ASSERT(gm->type == GGML_TYPE_F32);
    GGML_ASSERT(gv->type == GGML_TYPE_F32);
    GGML_ASSERT(p->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->ಬಫರ್ != ಶೂನ್ಯಸೂಚಕ);
    GGML_ASSERT(ggml_is_contiguous(x));
    GGML_ASSERT(ggml_is_contiguous(g));
    GGML_ASSERT(ggml_is_contiguous(gm));
    GGML_ASSERT(ggml_is_contiguous(gv));
    GGML_ASSERT(ggml_is_contiguous(p));
    GGML_ASSERT(ggml_are_same_shape(x, g));
    GGML_ASSERT(ggml_are_same_shape(x, gm));
    GGML_ASSERT(ggml_are_same_shape(x, gv));
    GGML_ASSERT(ggml_nelements(p) == 7);

    vk_pipeline pipeline = ggml_vk_op_get_pipeline(ctx, g, gm, gv, dst, GGML_OP_OPT_ಹೆಜ್ಜೆ_ADAMW);
    GGML_ASSERT(pipeline != ಶೂನ್ಯಸೂಚಕ);

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

ಶೂನ್ಯ ggml_vk_opt_ಹೆಜ್ಜೆ_adamw(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst) {
    ಸ್ಥಿರಾಂಕ size_t n = ggml_nelements(dst->src[0]);

    ggml_vk_op_f32_opt_ಹೆಜ್ಜೆ_adamw(
        ctx, subctx, dst,
        { (uint32_t)n, 0, 0.0f, 0.0f, 0.0f, 0.0f }
    );
}

ಶೂನ್ಯ ggml_vk_opt_ಹೆಜ್ಜೆ_sgd(ggml_backend_vk_context * ctx, vk_context& subctx, ಸ್ಥಿರಾಂಕ ggml_tensor * src0, ಸ್ಥಿರಾಂಕ ggml_tensor * src1, ಸ್ಥಿರಾಂಕ ggml_tensor * src2, ggml_tensor * dst) {
    ಸ್ಥಿರಾಂಕ size_t n = ggml_nelements(dst->src[0]);

    ggml_vk_op_f32<vk_op_ತಳ್ಳು_constants>(ctx, subctx, src0, src1, src2, ಶೂನ್ಯಸೂಚಕ, dst, GGML_OP_OPT_ಹೆಜ್ಜೆ_SGD, { (uint32_t)n, 0, 0.0f, 0.0f, 0.0f, 0.0f });
}

ಶೂನ್ಯ ggml_vk_concat(ggml_backend_vk_context * ctx, vk_context& subctx, ಸ್ಥಿರಾಂಕ ggml_tensor * src0, ಸ್ಥಿರಾಂಕ ggml_tensor * src1, ggml_tensor * dst) {
    ಪೂರ್ಣಾಂಕ * op_params = (ಪೂರ್ಣಾಂಕ *)dst->op_params;

    ಸ್ಥಿರಾಂಕ uint32_t unit_ಗಾತ್ರ = ggml_vk_concat_unit_ಗಾತ್ರ(dst->type);
    ಸ್ಥಿರಾಂಕ uint32_t units_ಪ್ರತಿ_ಖಂಡ = ggml_type_ಗಾತ್ರ(dst->type) / unit_ಗಾತ್ರ;
    ಸ್ಥಿರಾಂಕ uint32_t ಖಂಡ_ಗಾತ್ರ_3 = ggml_blck_ಗಾತ್ರ(dst->type);
    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ quantized = ggml_is_quantized(dst->type);

    // Address dimension 0 in packed storage units; higher strides may be noncontiguous.
    ಸ್ಥಿರಾಂಕ uint32_t ne00 = src0->ne[0] / ಖಂಡ_ಗಾತ್ರ_3 * units_ಪ್ರತಿ_ಖಂಡ;
    ಸ್ಥಿರಾಂಕ uint32_t ne10 = src1->ne[0] / ಖಂಡ_ಗಾತ್ರ_3 * units_ಪ್ರತಿ_ಖಂಡ;
    ಸ್ಥಿರಾಂಕ uint32_t ne20 =  dst->ne[0] / ಖಂಡ_ಗಾತ್ರ_3 * units_ಪ್ರತಿ_ಖಂಡ;
    ಸ್ಥಿರಾಂಕ uint32_t nb00 = quantized ? 1 : src0->nb[0] / unit_ಗಾತ್ರ;
    ಸ್ಥಿರಾಂಕ uint32_t nb10 = quantized ? 1 : src1->nb[0] / unit_ಗಾತ್ರ;
    ಸ್ಥಿರಾಂಕ uint32_t nb20 = quantized ? 1 :  dst->nb[0] / unit_ಗಾತ್ರ;

    vk_op_concat_ತಳ್ಳು_constants pc {{
        ne20 * (uint32_t)dst->ne[1] * (uint32_t)dst->ne[2] * (uint32_t)dst->ne[3],
        ne00, (uint32_t)src0->ne[1], (uint32_t)src0->ne[2],(uint32_t)src0->ne[3], nb00, (uint32_t)src0->nb[1] / unit_ಗಾತ್ರ, (uint32_t)src0->nb[2] / unit_ಗಾತ್ರ, (uint32_t)src0->nb[3] / unit_ಗಾತ್ರ,
        ne10, (uint32_t)src1->ne[1], (uint32_t)src1->ne[2],(uint32_t)src1->ne[3], nb10, (uint32_t)src1->nb[1] / unit_ಗಾತ್ರ, (uint32_t)src1->nb[2] / unit_ಗಾತ್ರ, (uint32_t)src1->nb[3] / unit_ಗಾತ್ರ,
        ne20, (uint32_t) dst->ne[1], (uint32_t) dst->ne[2],(uint32_t) dst->ne[3], nb20, (uint32_t) dst->nb[1] / unit_ಗಾತ್ರ, (uint32_t) dst->nb[2] / unit_ಗಾತ್ರ, (uint32_t) dst->nb[3] / unit_ಗಾತ್ರ,
        0,
        0.0f, 0.0f, op_params[0],
    }};
    ggml_vk_op_f32<vk_op_concat_ತಳ್ಳು_constants>(ctx, subctx, src0, src1, ಶೂನ್ಯಸೂಚಕ, ಶೂನ್ಯಸೂಚಕ, dst, GGML_OP_CONCAT, std::move(pc));
}

ಶೂನ್ಯ ggml_vk_upscale(ggml_backend_vk_context * ctx, vk_context& subctx, ಸ್ಥಿರಾಂಕ ggml_tensor * src0, ggml_tensor * dst) {
    ಸ್ಥಿರಾಂಕ uint32_t src0_type_ಗಾತ್ರ = ggml_type_ಗಾತ್ರ(src0->type);
    ಸ್ಥಿರಾಂಕ uint32_t mode = (uint32_t)ggml_get_op_params_i32(dst, 0);

    GGML_TENSOR_UNARY_OP_LOCALS

    ದಶಮಾಂಶ sf0 = (ದಶಮಾಂಶ)ne0 / ne00;
    ದಶಮಾಂಶ sf1 = (ದಶಮಾಂಶ)ne1 / ne01;
    ದಶಮಾಂಶ sf2 = (ದಶಮಾಂಶ)ne2 / ne02;
    ದಶಮಾಂಶ sf3 = (ದಶಮಾಂಶ)ne3 / ne03;
    ದಶಮಾಂಶ pixel_ಸ್ಥಾನಾಂತರ = 0.5f;

    ಆದರೆ (mode & GGML_SCALE_FLAG_ALIGN_CORNERS) {
        sf0 = ne0 > 1 && ne00 > 1 ? (ದಶಮಾಂಶ)(ne0 - 1) / (ne00 - 1) : sf0;
        sf1 = ne1 > 1 && ne01 > 1 ? (ದಶಮಾಂಶ)(ne1 - 1) / (ne01 - 1) : sf1;
        pixel_ಸ್ಥಾನಾಂತರ = 0.0f;
    }

    ggml_vk_op_f32<vk_op_upscale_ತಳ್ಳು_constants>(ctx, subctx, src0, ಶೂನ್ಯಸೂಚಕ, ಶೂನ್ಯಸೂಚಕ, ಶೂನ್ಯಸೂಚಕ, dst, GGML_OP_UPSCALE, {
        (uint32_t)ggml_nelements(dst), 0, 0,
        (uint32_t)ne00, (uint32_t)ne01,
        (uint32_t)nb00 / src0_type_ಗಾತ್ರ, (uint32_t)nb01 / src0_type_ಗಾತ್ರ, (uint32_t)nb02 / src0_type_ಗಾತ್ರ, (uint32_t)nb03 / src0_type_ಗಾತ್ರ,
        (uint32_t)ne0, (uint32_t)ne1, (uint32_t)ne2, (uint32_t)ne3,
        sf0, sf1, sf2, sf3, pixel_ಸ್ಥಾನಾಂತರ
    });
}

ಶೂನ್ಯ ggml_vk_scale(ggml_backend_vk_context * ctx, vk_context& subctx, ಸ್ಥಿರಾಂಕ ggml_tensor * src0, ggml_tensor * dst) {
    vk_op_unary_ತಳ್ಳು_constants p = vk_op_unary_ತಳ್ಳು_constants_init(src0, dst);
    p.param1 = ggml_get_op_params_f32(dst, 0);
    p.param2 = ggml_get_op_params_f32(dst, 1);

    ggml_vk_op_f32(ctx, subctx, src0, ಶೂನ್ಯಸೂಚಕ, ಶೂನ್ಯಸೂಚಕ, ಶೂನ್ಯಸೂಚಕ, dst, GGML_OP_SCALE, std::move(p));
}

ಶೂನ್ಯ ggml_vk_sqr(ggml_backend_vk_context * ctx, vk_context& subctx, ಸ್ಥಿರಾಂಕ ggml_tensor * src0, ggml_tensor * dst) {
    ggml_vk_op_f32(ctx, subctx, src0, ಶೂನ್ಯಸೂಚಕ, ಶೂನ್ಯಸೂಚಕ, ಶೂನ್ಯಸೂಚಕ, dst, GGML_OP_SQR, vk_op_unary_ತಳ್ಳು_constants_init(src0, dst));
}

ಶೂನ್ಯ ggml_vk_sqrt(ggml_backend_vk_context * ctx, vk_context& subctx, ಸ್ಥಿರಾಂಕ ggml_tensor * src0, ggml_tensor * dst) {
    ggml_vk_op_f32(ctx, subctx, src0, ಶೂನ್ಯಸೂಚಕ, ಶೂನ್ಯಸೂಚಕ, ಶೂನ್ಯಸೂಚಕ, dst, GGML_OP_SQRT, vk_op_unary_ತಳ್ಳು_constants_init(src0, dst));
}

ಶೂನ್ಯ ggml_vk_add1(ggml_backend_vk_context * ctx, vk_context& subctx, ಸ್ಥಿರಾಂಕ ggml_tensor * src0, ಸ್ಥಿರಾಂಕ ggml_tensor * src1, ggml_tensor * dst) {
    ಸ್ಥಿರಾಂಕ uint32_t src0_type_ಗಾತ್ರ = ggml_type_ಗಾತ್ರ(src0->type);
    ಸ್ಥಿರಾಂಕ uint32_t src1_type_ಗಾತ್ರ = ggml_type_ಗಾತ್ರ(src1->type);
    ಸ್ಥಿರಾಂಕ uint32_t dst_type_ಗಾತ್ರ = ggml_type_ಗಾತ್ರ(dst->type);

    ggml_vk_op_f32<vk_op_ದ್ವಿಮಾನ_ತಳ್ಳು_constants>(ctx, subctx, src0, src1, ಶೂನ್ಯಸೂಚಕ, ಶೂನ್ಯಸೂಚಕ, dst, GGML_OP_ADD1, {
        (uint32_t)ggml_nelements(src0),
        (uint32_t)src0->ne[0], (uint32_t)src0->ne[1], (uint32_t)src0->ne[2],(uint32_t)src0->ne[3], (uint32_t)src0->nb[0] / src0_type_ಗಾತ್ರ, (uint32_t)src0->nb[1] / src0_type_ಗಾತ್ರ, (uint32_t)src0->nb[2] / src0_type_ಗಾತ್ರ, (uint32_t)src0->nb[3] / src0_type_ಗಾತ್ರ,
        (uint32_t)src1->ne[0], (uint32_t)src1->ne[1], (uint32_t)src1->ne[2],(uint32_t)src1->ne[3], (uint32_t)src1->nb[0] / src1_type_ಗಾತ್ರ, (uint32_t)src1->nb[1] / src1_type_ಗಾತ್ರ, (uint32_t)src1->nb[2] / src1_type_ಗಾತ್ರ, (uint32_t)src1->nb[3] / src1_type_ಗಾತ್ರ,
        (uint32_t) dst->ne[0], (uint32_t) dst->ne[1], (uint32_t) dst->ne[2],(uint32_t) dst->ne[3], (uint32_t) dst->nb[0] /  dst_type_ಗಾತ್ರ, (uint32_t) dst->nb[1] /  dst_type_ಗಾತ್ರ, (uint32_t) dst->nb[2] /  dst_type_ಗಾತ್ರ, (uint32_t) dst->nb[3] /  dst_type_ಗಾತ್ರ,
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

    vk_pipeline pipeline = ggml_vk_op_get_pipeline(ctx, ಶೂನ್ಯಸೂಚಕ, ಶೂನ್ಯಸೂಚಕ, ಶೂನ್ಯಸೂಚಕ, dst, GGML_OP_ARANGE);
    GGML_ASSERT(pipeline != ಶೂನ್ಯಸೂಚಕ);

    ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);
    vk_subbuffer dst_buf = ggml_vk_tensor_subbuffer(ctx, dst, ಸುಳ್ಳು);

    std::array<uint32_t, 3> elements = { (uint32_t)ggml_nelements(dst), 1, 1 };

    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { dst_buf }, pc, elements);
}

ಶೂನ್ಯ ggml_vk_fill(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst) {
    VK_LOG_DEBUG("ggml_vk_fill(dst=" << dst << ", ne=" << ggml_nelements(dst) << ")");
    ಸ್ಥಿರಾಂಕ uint64_t n = ggml_nelements(dst);
    GGML_ASSERT(n > 0);

    vk_op_ತಳ್ಳು_constants pc = {
        (uint32_t)n,
        1,
        ggml_get_op_params_f32(dst, 0),
        0.0f,
        0.0f, 0.0f,
    };

    vk_pipeline pipeline = ggml_vk_op_get_pipeline(ctx, ಶೂನ್ಯಸೂಚಕ, ಶೂನ್ಯಸೂಚಕ, ಶೂನ್ಯಸೂಚಕ, dst, GGML_OP_FILL);
    GGML_ASSERT(pipeline != ಶೂನ್ಯಸೂಚಕ);

    // Split the task distribution to 2D to avoid exceeding maxComputeWorkGroupCount
    ಸ್ಥಿರಾಂಕ uint32_t ಒಟ್ಟು_wg = CEIL_DIV(n, pipeline->wg_denoms[0]);
    ಸ್ಥಿರಾಂಕ uint32_t wg_x = std::min(ಒಟ್ಟು_wg, ctx->device->properties.limits.max_Compute_Work_ಗುಂಪು_ಎಣಿಕೆ[0]);
    ಸ್ಥಿರಾಂಕ uint32_t wg_y = CEIL_DIV(ಒಟ್ಟು_wg, wg_x);
    GGML_ASSERT(wg_y <= ctx->device->properties.limits.max_Compute_Work_ಗುಂಪು_ಎಣಿಕೆ[1]);

    ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);
    vk_subbuffer dst_buf = ggml_vk_tensor_subbuffer(ctx, dst, ಸುಳ್ಳು);

    std::array<uint32_t, 3> elements = { wg_x * pipeline->wg_denoms[0], wg_y * pipeline->wg_denoms[1], 1 };
    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { dst_buf }, pc, elements);
}

ಶೂನ್ಯ ggml_vk_sin(ggml_backend_vk_context * ctx, vk_context& subctx, ಸ್ಥಿರಾಂಕ ggml_tensor * src0, ggml_tensor * dst) {
    ggml_vk_op_f32(ctx, subctx, src0, ಶೂನ್ಯಸೂಚಕ, ಶೂನ್ಯಸೂಚಕ, ಶೂನ್ಯಸೂಚಕ, dst, GGML_OP_SIN, vk_op_unary_ತಳ್ಳು_constants_init(src0, dst));
}

ಶೂನ್ಯ ggml_vk_cos(ggml_backend_vk_context * ctx, vk_context& subctx, ಸ್ಥಿರಾಂಕ ggml_tensor * src0, ggml_tensor * dst) {
    ggml_vk_op_f32(ctx, subctx, src0, ಶೂನ್ಯಸೂಚಕ, ಶೂನ್ಯಸೂಚಕ, ಶೂನ್ಯಸೂಚಕ, dst, GGML_OP_COS, vk_op_unary_ತಳ್ಳು_constants_init(src0, dst));
}

ಶೂನ್ಯ ggml_vk_log(ggml_backend_vk_context * ctx, vk_context& subctx, ಸ್ಥಿರಾಂಕ ggml_tensor * src0, ggml_tensor * dst) {
    ggml_vk_op_f32(ctx, subctx, src0, ಶೂನ್ಯಸೂಚಕ, ಶೂನ್ಯಸೂಚಕ, ಶೂನ್ಯಸೂಚಕ, dst, GGML_OP_LOG, vk_op_unary_ತಳ್ಳು_constants_init(src0, dst));
}

ಶೂನ್ಯ ggml_vk_tri(ggml_backend_vk_context * ctx, vk_context& subctx, ಸ್ಥಿರಾಂಕ ggml_tensor * src0, ggml_tensor * dst) {
    vk_op_unary_ತಳ್ಳು_constants p = vk_op_unary_ತಳ್ಳು_constants_init(src0, dst);
    p.param1 = ggml_get_op_params_f32(dst, 0);

    ggml_vk_op_f32(ctx, subctx, src0, ಶೂನ್ಯಸೂಚಕ, ಶೂನ್ಯಸೂಚಕ, ಶೂನ್ಯಸೂಚಕ, dst, GGML_OP_TRI, std::move(p));
}

ಶೂನ್ಯ ggml_vk_diag(ggml_backend_vk_context * ctx, vk_context& subctx, ಸ್ಥಿರಾಂಕ ggml_tensor * src0, ggml_tensor * dst) {
    vk_op_unary_ತಳ್ಳು_constants p = vk_op_unary_ತಳ್ಳು_constants_init(src0, dst, ggml_nelements(dst));

    ggml_vk_op_f32(ctx, subctx, src0, ಶೂನ್ಯಸೂಚಕ, ಶೂನ್ಯಸೂಚಕ, ಶೂನ್ಯಸೂಚಕ, dst, GGML_OP_DIAG, std::move(p));
}

ಶೂನ್ಯ ggml_vk_clamp(ggml_backend_vk_context * ctx, vk_context& subctx, ಸ್ಥಿರಾಂಕ ggml_tensor * src0, ggml_tensor * dst) {
    vk_op_unary_ತಳ್ಳು_constants p = vk_op_unary_ತಳ್ಳು_constants_init(src0, dst);
    p.param1 = ggml_get_op_params_f32(dst, 0);
    p.param2 = ggml_get_op_params_f32(dst, 1);

    ggml_vk_op_f32(ctx, subctx, src0, ಶೂನ್ಯಸೂಚಕ, ಶೂನ್ಯಸೂಚಕ, ಶೂನ್ಯಸೂಚಕ, dst, GGML_OP_CLAMP, std::move(p));
}

ಶೂನ್ಯ ggml_vk_pad(ggml_backend_vk_context * ctx, vk_context& subctx, ಸ್ಥಿರಾಂಕ ggml_tensor * src0, ggml_tensor * dst) {
    vk_op_pad_ತಳ್ಳು_constants p = vk_op_pad_ತಳ್ಳು_constants_init(src0, dst);
    ggml_vk_op_f32(ctx, subctx, src0, ಶೂನ್ಯಸೂಚಕ, ಶೂನ್ಯಸೂಚಕ, ಶೂನ್ಯಸೂಚಕ, dst, GGML_OP_PAD, std::move(p));
}

ಶೂನ್ಯ ggml_vk_pad_reflect_1d(ggml_backend_vk_context * ctx, vk_context& subctx, ಸ್ಥಿರಾಂಕ ggml_tensor * src0, ggml_tensor * dst) {
    ಸ್ಥಿರಾಂಕ uint32_t p0 = (uint32_t)dst->op_params[0];
    ಸ್ಥಿರಾಂಕ uint32_t p1 = (uint32_t)dst->op_params[1];

    vk_op_unary_ತಳ್ಳು_constants p = vk_op_unary_ತಳ್ಳು_constants_init(src0, dst, ggml_nelements(dst));
    ಸ್ಮರಣೆ_ನಕಲು(&p.param1, &p0, ಗಾತ್ರ(ದಶಮಾಂಶ));
    ಸ್ಮರಣೆ_ನಕಲು(&p.param2, &p1, ಗಾತ್ರ(ದಶಮಾಂಶ));

    ggml_vk_op_f32(ctx, subctx, src0, ಶೂನ್ಯಸೂಚಕ, ಶೂನ್ಯಸೂಚಕ, ಶೂನ್ಯಸೂಚಕ, dst, GGML_OP_PAD_REFLECT_1D, std::move(p));
}

ಶೂನ್ಯ ggml_vk_roll(ggml_backend_vk_context * ctx, vk_context& subctx, ಸ್ಥಿರಾಂಕ ggml_tensor * src0, ggml_tensor * dst) {
    ಸ್ಥಿರಾಂಕ int32_t s0 = ggml_get_op_params_i32(dst, 0);
    ಸ್ಥಿರಾಂಕ int32_t s1 = ggml_get_op_params_i32(dst, 1);
    ಸ್ಥಿರಾಂಕ int32_t s2 = ggml_get_op_params_i32(dst, 2);
    ಸ್ಥಿರಾಂಕ int32_t s3 = ggml_get_op_params_i32(dst, 3);
    ಸ್ಥಿರಾಂಕ uint32_t s01_packed = ((s0 + 0x8000) << 16) | (s1 + 0x8000);
    ಸ್ಥಿರಾಂಕ uint32_t s23_packed = ((s2 + 0x8000) << 16) | (s3 + 0x8000);

    vk_op_unary_ತಳ್ಳು_constants p = vk_op_unary_ತಳ್ಳು_constants_init(src0, dst);
    ಸ್ಮರಣೆ_ನಕಲು(&p.param1, &s01_packed, ಗಾತ್ರ(ದಶಮಾಂಶ));
    ಸ್ಮರಣೆ_ನಕಲು(&p.param2, &s23_packed, ಗಾತ್ರ(ದಶಮಾಂಶ));

    ggml_vk_op_f32(ctx, subctx, src0, ಶೂನ್ಯಸೂಚಕ, ಶೂನ್ಯಸೂಚಕ, ಶೂನ್ಯಸೂಚಕ, dst, GGML_OP_ROLL, std::move(p));
}

ಶೂನ್ಯ ggml_vk_repeat(ggml_backend_vk_context * ctx, vk_context& subctx, ಸ್ಥಿರಾಂಕ ggml_tensor * src0, ggml_tensor * dst) {
    vk_op_unary_ತಳ್ಳು_constants p = vk_op_unary_ತಳ್ಳು_constants_init(src0, dst, ggml_nelements(dst));
    ggml_vk_op_f32(ctx, subctx, src0, ಶೂನ್ಯಸೂಚಕ, ಶೂನ್ಯಸೂಚಕ, ಶೂನ್ಯಸೂಚಕ, dst, GGML_OP_REPEAT, std::move(p));
}

ಶೂನ್ಯ ggml_vk_repeat_ಹಿಂಭಾಗ(ggml_backend_vk_context * ctx, vk_context& subctx, ಸ್ಥಿರಾಂಕ ggml_tensor * src0, ggml_tensor * dst) {
    vk_op_unary_ತಳ್ಳು_constants p = vk_op_unary_ತಳ್ಳು_constants_init(src0, dst, ggml_nelements(dst));
    ggml_vk_op_f32(ctx, subctx, src0, ಶೂನ್ಯಸೂಚಕ, ಶೂನ್ಯಸೂಚಕ, ಶೂನ್ಯಸೂಚಕ, dst, GGML_OP_REPEAT_ಹಿಂಭಾಗ, std::move(p));
}

ಶೂನ್ಯ ggml_vk_cpy(ggml_backend_vk_context * ctx, vk_context& subctx, ಸ್ಥಿರಾಂಕ ggml_tensor * src0, ggml_tensor * dst) {
    uint32_t ne = (uint32_t)ggml_nelements(src0);
    ಆದರೆ (ggml_is_quantized(src0->type) && ggml_is_quantized(dst->type)) {
        // Convert from number of logical elements to 2- or 4-byte units.
        ne /= ggml_blck_ಗಾತ್ರ(src0->type);
        ಆದರೆ ((ggml_type_ಗಾತ್ರ(src0->type) % 4) == 0) {
            ne *= ggml_type_ಗಾತ್ರ(src0->type) / 4;
        } ಇಲ್ಲದಿದ್ದರೆ {
            ne *= ggml_type_ಗಾತ್ರ(src0->type) / 2;
        }
    }

    vk_op_unary_ತಳ್ಳು_constants p = vk_op_unary_ತಳ್ಳು_constants_init(src0, dst, ne);
    ggml_vk_op_f32(ctx, subctx, src0, ಶೂನ್ಯಸೂಚಕ, ಶೂನ್ಯಸೂಚಕ, ಶೂನ್ಯಸೂಚಕ, dst, GGML_OP_CPY, std::move(p));
}

ಶೂನ್ಯ ggml_vk_ಗಣ_ಅಡ್ಡಸಾಲುಗಳು(ggml_backend_vk_context * ctx, vk_context& subctx, ಸ್ಥಿರಾಂಕ ggml_tensor * src0, ಸ್ಥಿರಾಂಕ ggml_tensor * src1, ggml_tensor * dst) {
    ಸ್ಥಿರಾಂಕ uint32_t src0_type_ಗಾತ್ರ = ggml_type_ಗಾತ್ರ(src0->type);
    ಸ್ಥಿರಾಂಕ uint32_t src1_type_ಗಾತ್ರ = ggml_type_ಗಾತ್ರ(src1->type);
    ಸ್ಥಿರಾಂಕ uint32_t dst_type_ಗಾತ್ರ = ggml_type_ಗಾತ್ರ(dst->type);

    // Skip empty skip_rows operations. For most ops the empty check at the start
    // of ggml_vk_build_graph is sufficient, but set_rows can have a nonempty dst
    // with empty srcs.
    ಆದರೆ (ggml_is_ಖಾಲಿಗಳು(src0) || ggml_is_ಖಾಲಿಗಳು(src1)) {
        ಹಿಂತಿರುಗಿ;
    }

    ggml_vk_op_f32<vk_op_ದ್ವಿಮಾನ_ತಳ್ಳು_constants>(ctx, subctx, src0, src1, ಶೂನ್ಯಸೂಚಕ, ಶೂನ್ಯಸೂಚಕ, dst, GGML_OP_ಗಣ_ಅಡ್ಡಸಾಲುಗಳು, {
        (uint32_t)ggml_nelements(src0),
        (uint32_t)src0->ne[0], (uint32_t)src0->ne[1], (uint32_t)src0->ne[2],(uint32_t)src0->ne[3], (uint32_t)src0->nb[0] / src0_type_ಗಾತ್ರ, (uint32_t)src0->nb[1] / src0_type_ಗಾತ್ರ, (uint32_t)src0->nb[2] / src0_type_ಗಾತ್ರ, (uint32_t)src0->nb[3] / src0_type_ಗಾತ್ರ,
        (uint32_t)src1->ne[0], (uint32_t)src1->ne[1], (uint32_t)src1->ne[2],(uint32_t)src1->ne[3], (uint32_t)src1->nb[0] / src1_type_ಗಾತ್ರ, (uint32_t)src1->nb[1] / src1_type_ಗಾತ್ರ, (uint32_t)src1->nb[2] / src1_type_ಗಾತ್ರ, (uint32_t)src1->nb[3] / src1_type_ಗಾತ್ರ,
        (uint32_t) dst->ne[0], (uint32_t) dst->ne[1], (uint32_t) dst->ne[2],(uint32_t) dst->ne[3], (uint32_t) dst->nb[0] /  dst_type_ಗಾತ್ರ, (uint32_t) dst->nb[1] /  dst_type_ಗಾತ್ರ, (uint32_t) dst->nb[2] /  dst_type_ಗಾತ್ರ, (uint32_t) dst->nb[3] /  dst_type_ಗಾತ್ರ,
        0,
        0.0f, 0.0f, 0,
    });
}

ಶೂನ್ಯ ggml_vk_silu_ಹಿಂಭಾಗ(ggml_backend_vk_context * ctx, vk_context& subctx, ಸ್ಥಿರಾಂಕ ggml_tensor * src0, ಸ್ಥಿರಾಂಕ ggml_tensor * src1, ggml_tensor * dst) {
    ggml_vk_op_f32<vk_op_ತಳ್ಳು_constants>(ctx, subctx, src0, src1, ಶೂನ್ಯಸೂಚಕ, ಶೂನ್ಯಸೂಚಕ, dst, GGML_OP_SILU_ಹಿಂಭಾಗ, { (uint32_t)ggml_nelements(src0), 0, 0.0f, 0.0f, 0.0f, 0.0f });
}

ಶೂನ್ಯ ggml_vk_norm(ggml_backend_vk_context * ctx, vk_context& subctx, ಸ್ಥಿರಾಂಕ ggml_tensor * src0, ggml_tensor * dst) {
    ದಶಮಾಂಶ * op_params = (ದಶಮಾಂಶ *)dst->op_params;
    vk_op_unary_ತಳ್ಳು_constants p = vk_op_unary_ತಳ್ಳು_constants_init(src0, dst);
    p.param1 = op_params[0];

    ggml_vk_op_f32(ctx, subctx, src0, ಶೂನ್ಯಸೂಚಕ, ಶೂನ್ಯಸೂಚಕ, ಶೂನ್ಯಸೂಚಕ, dst, GGML_OP_NORM, std::move(p));
}

ಶೂನ್ಯ ggml_vk_ಗುಂಪು_norm(ggml_backend_vk_context * ctx, vk_context& subctx, ಸ್ಥಿರಾಂಕ ggml_tensor * src0, ggml_tensor * dst) {
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ * int_op_params = (ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ *)dst->op_params;
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * float_op_params = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *)dst->op_params;

    ಸ್ಥಿರಾಂಕ uint32_t ಅಂಕೆ_groups = int_op_params[0];
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ eps = float_op_params[1];
    ಸ್ಥಿರಾಂಕ uint32_t ಗುಂಪು_ಗಾತ್ರ = src0->ne[0] * src0->ne[1] * ((src0->ne[2] + ಅಂಕೆ_groups - 1) / ಅಂಕೆ_groups);

    ggml_vk_op_f32<vk_op_ತಳ್ಳು_constants>(ctx, subctx, src0, ಶೂನ್ಯಸೂಚಕ, ಶೂನ್ಯಸೂಚಕ, ಶೂನ್ಯಸೂಚಕ, dst, GGML_OP_ಗುಂಪು_NORM, { ಗುಂಪು_ಗಾತ್ರ, 0, eps, 0.0f, 0.0f, 0.0f });
}

ಸ್ಥಿರ uint32_t ggml_vk_rms_ಅಂಕೆ_partials(ggml_backend_vk_context * ctx, ಸ್ಥಿರಾಂಕ ggml_tensor *ಗಂಟು) {
    ಸ್ಥಿರಾಂಕ uint32_t ne = (uint32_t)ಗಂಟು->ne[0];
    ಸ್ಥಿರಾಂಕ uint32_t denom = ctx->device->pipeline_ಕೂಡಿಸು_rms[0][0][0]->wg_denoms[0];
    ಸ್ಥಿರಾಂಕ uint32_t ಅಂಕೆ_partials = CEIL_DIV(ne, denom);
    ಹಿಂತಿರುಗಿ ಅಂಕೆ_partials;
}

uint32_t ggml_vk_rms_partials_ಗಾತ್ರ(ggml_backend_vk_context * ctx, ಸ್ಥಿರಾಂಕ ggml_tensor *ಗಂಟು) {
    ಸ್ಥಿರಾಂಕ uint32_t ಅಂಕೆ_partials = ggml_vk_rms_ಅಂಕೆ_partials(ctx, ಗಂಟು);
    ಸ್ಥಿರಾಂಕ uint32_t ಅಂಕೆ_ಬೈಟುಗಳು = ROUNDUP_POW2(ಅಂಕೆ_partials * ಗಾತ್ರ(uint32_t), ctx->device->partials_binding_alignment);
    ಹಿಂತಿರುಗಿ ಅಂಕೆ_ಬೈಟುಗಳು;
}

ಸ್ಥಿರ vk_op_rope_ತಳ್ಳು_constants ggml_vk_make_rope_constants(ಸ್ಥಿರಾಂಕ ggml_tensor *dst, ಸ್ಥಿರಾಂಕ ggml_tensor *src0, ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ has_ff, ತಾರ್ಕಿಕ backprop, ಸ್ಥಿರಾಂಕ uint32_t ಗಣ_ಅಡ್ಡಸಾಲುಗಳು_stride) {
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ n_dims        = ((ಸ್ಥಿರಾಂಕ int32_t *) dst->op_params)[1];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ mode          = ((ಸ್ಥಿರಾಂಕ int32_t *) dst->op_params)[2];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ n_offs        = ((ಸ್ಥಿರಾಂಕ int32_t *) dst->op_params)[15];
    // const int n_ctx         = ((const int32_t *) dst->op_params)[3];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ n_ctx_orig    = ((ಸ್ಥಿರಾಂಕ int32_t *) dst->op_params)[4];
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ freq_ತಳ   = ((ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *)   dst->op_params)[5];
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ freq_scale  = ((ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *)   dst->op_params)[6];
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ ext_ಗುಣಕ  = ((ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *)   dst->op_params)[7];
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ attn_ಗುಣಕ = ((ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *)   dst->op_params)[8];
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ beta_fast   = ((ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *)   dst->op_params)[9];
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ beta_slow   = ((ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *)   dst->op_params)[10];
    ಪೂರ್ಣಾಂಕ sections[4] {};
    ಆದರೆ (mode & GGML_ROPE_TYPE_MROPE) {
        ಸ್ಮರಣೆ_ನಕಲು(sections, (ಸ್ಥಿರಾಂಕ int32_t *) dst->op_params + 11, ಗಾತ್ರ(ಪೂರ್ಣಾಂಕ)*4);
    }

    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ is_imrope = mode == GGML_ROPE_TYPE_IMROPE;

    ದಶಮಾಂಶ corr_dims[2];
    ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_ತಳ, beta_fast, beta_slow, corr_dims);

    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ theta_scale = powf(freq_ತಳ, -2.0f/n_dims);

    uint32_t nb01 = src0->nb[1] / ggml_type_ಗಾತ್ರ(src0->type);
    uint32_t nb02 = src0->nb[2] / ggml_type_ಗಾತ್ರ(src0->type);
    uint32_t nb03 = src0->nb[3] / ggml_type_ಗಾತ್ರ(src0->type);

    uint32_t nb11 = dst->nb[1] / ggml_type_ಗಾತ್ರ(dst->type);
    uint32_t nb12 = dst->nb[2] / ggml_type_ಗಾತ್ರ(dst->type);
    uint32_t nb13 = dst->nb[3] / ggml_type_ಗಾತ್ರ(dst->type);

    vk_op_rope_ತಳ್ಳು_constants rope {
        (uint32_t)mode, (uint32_t)ggml_nrows(src0), (uint32_t)n_dims, (uint32_t)n_offs, freq_scale,
        freq_ತಳ, ext_ಗುಣಕ, attn_ಗುಣಕ, {corr_dims[0], corr_dims[1]}, theta_scale, has_ff,
        { sections[0], sections[1], sections[2], sections[3] }, is_imrope, backprop, ಗಣ_ಅಡ್ಡಸಾಲುಗಳು_stride,

        (uint32_t)src0->ne[0],
        (uint32_t)src0->ne[1],
        (uint32_t)src0->ne[2],
        nb01, nb02, nb03,
        nb11, nb12, nb13,
        0, 0, // a_offset, d_offset filled in by init_pushconst_tensor_offsets
    };

    ಹಿಂತಿರುಗಿ rope;
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_vk_rms_norm_ಮುಕ್ತಾಯ(ggml_backend_vk_context * ctx, ಸ್ಥಿರಾಂಕ ggml_tensor * src0) {
    ಆದರೆ (ctx->do_ಕೂಡಿಸು_rms_partials_ಸ್ಥಾನಾಂತರ_calculation) {
        ctx->prealloc_ಗಾತ್ರ_ಕೂಡಿಸು_rms_partials_ಸ್ಥಾನಾಂತರ += ggml_vk_rms_partials_ಗಾತ್ರ(ctx, src0);
        ctx->do_ಕೂಡಿಸು_rms_partials = ಸುಳ್ಳು;
        ctx->do_ಕೂಡಿಸು_rms_partials_ಸ್ಥಾನಾಂತರ_calculation = ಸುಳ್ಳು;
    }
}

ಶೂನ್ಯ ggml_vk_rms_norm(ggml_backend_vk_context * ctx, vk_context& subctx, ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_cgraph * cgraph, ಪೂರ್ಣಾಂಕ ಗಂಟು_idx, ದಶಮಾಂಶ * op_params) {
    ggml_tensor * rms = cgraph->nodes[ಗಂಟು_idx];
    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = rms->src[0];

    ಆದರೆ (ctx->fused_rms_norm_ಬಹುಲಕ == RMS_NORM_VIEW_ಗಣ_ಅಡ್ಡಸಾಲುಗಳು) {
        GGML_ASSERT(ctx->ಅಂಕೆ_additional_fused_ops == 2);
        ggml_tensor * ಗಣ_ಅಡ್ಡಸಾಲುಗಳು = cgraph->nodes[ಗಂಟು_idx + 2];
        ಸ್ಥಿರಾಂಕ ggml_tensor * indices = ಗಣ_ಅಡ್ಡಸಾಲುಗಳು->src[1];
        vk_op_ದ್ವಿಮಾನ_ತಳ್ಳು_constants pc = ggml_vk_rms_norm_ತಳ್ಳು_constants(src0, src0, ಗಣ_ಅಡ್ಡಸಾಲುಗಳು, op_params[0], 0);
        init_pushconst_tensor_offsets(ctx, pc, src0, src0, ಶೂನ್ಯಸೂಚಕ, ಶೂನ್ಯಸೂಚಕ, ಗಣ_ಅಡ್ಡಸಾಲುಗಳು);

        vk_pipeline pipeline = ಗಣ_ಅಡ್ಡಸಾಲುಗಳು->type == GGML_TYPE_F16 ?
            ctx->device->pipeline_rms_norm_ಗಣ_ಅಡ್ಡಸಾಲುಗಳು_f32_f16 : ctx->device->pipeline_rms_norm_ಗಣ_ಅಡ್ಡಸಾಲುಗಳು_f32_f32;
        ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);
        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline,
            {
                ggml_vk_tensor_subbuffer(ctx, src0, ನಿಜ),
                ggml_vk_tensor_subbuffer(ctx, src0, ನಿಜ),
                ggml_vk_tensor_subbuffer(ctx, ಗಣ_ಅಡ್ಡಸಾಲುಗಳು, ನಿಜ),
                ggml_vk_tensor_subbuffer(ctx, indices),
            }, pc, {
                (uint32_t)src0->ne[1],
                std::min((uint32_t)src0->ne[2], ctx->device->properties.limits.max_Compute_Work_ಗುಂಪು_ಎಣಿಕೆ[1]),
                std::min((uint32_t)src0->ne[3], ctx->device->properties.limits.max_Compute_Work_ಗುಂಪು_ಎಣಿಕೆ[2]),
            });
        ggml_vk_rms_norm_ಮುಕ್ತಾಯ(ctx, src0);
        ಹಿಂತಿರುಗಿ;
    }

    ಆದರೆ (ctx->fused_rms_norm_ಬಹುಲಕ == RMS_NORM_MUL_ಕೂಡಿಸು || ctx->fused_rms_norm_ಬಹುಲಕ == RMS_NORM_MUL_ಕೂಡಿಸು_MUL) {
        ggml_tensor * mul = cgraph->nodes[ಗಂಟು_idx + 1];
        ggml_tensor * ಕೂಡಿಸು_2 = cgraph->nodes[ಗಂಟು_idx + 2];
        ಸ್ಥಿರಾಂಕ ggml_tensor * ತೂಕ = mul->src[0] == rms ? mul->src[1] : mul->src[0];
        ಸ್ಥಿರಾಂಕ ggml_tensor * residual = ಕೂಡಿಸು_2->src[0] == mul ? ಕೂಡಿಸು_2->src[1] : ಕೂಡಿಸು_2->src[0];
        ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ do_post_multiply = ctx->fused_rms_norm_ಬಹುಲಕ == RMS_NORM_MUL_ಕೂಡಿಸು_MUL;
        GGML_ASSERT(ctx->ಅಂಕೆ_additional_fused_ops == (do_post_multiply ? 3 : 2));
        ggml_tensor * dst = do_post_multiply ? cgraph->nodes[ಗಂಟು_idx + 3] : ಕೂಡಿಸು_2;
        ಸ್ಥಿರಾಂಕ ggml_tensor * post_scale = do_post_multiply ?
            (dst->src[0] == ಕೂಡಿಸು_2 ? dst->src[1] : dst->src[0]) : src0;

        ಸ್ಥಿರಾಂಕ uint32_t ಅಂಕೆ_partials = ctx->do_ಕೂಡಿಸು_rms_partials ? ggml_vk_rms_ಅಂಕೆ_partials(ctx, dst) : 0;
        vk_op_ದ್ವಿಮಾನ_ತಳ್ಳು_constants pc = ggml_vk_rms_norm_ತಳ್ಳು_constants(src0, ತೂಕ, dst, op_params[0], ಅಂಕೆ_partials);
        init_pushconst_tensor_offsets(ctx, pc, src0, ತೂಕ, residual, post_scale, dst);

        vk_pipeline pipeline;
        ಆದರೆ (ctx->do_ಕೂಡಿಸು_rms_partials) {
            pipeline = do_post_multiply ?
                ctx->device->pipeline_rms_norm_mul_ಕೂಡಿಸು_mul_partials_f32 : ctx->device->pipeline_rms_norm_mul_ಕೂಡಿಸು_partials_f32;
        } ಇಲ್ಲದಿದ್ದರೆ {
            pipeline = do_post_multiply ?
                ctx->device->pipeline_rms_norm_mul_ಕೂಡಿಸು_mul_f32 : ctx->device->pipeline_rms_norm_mul_ಕೂಡಿಸು_f32;
        }
        ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);
        ಆದರೆ (ctx->do_ಕೂಡಿಸು_rms_partials) {
            ggml_vk_dispatch_pipeline(ctx, subctx, pipeline,
                {
                    ggml_vk_tensor_subbuffer(ctx, src0, ನಿಜ),
                    ggml_vk_tensor_subbuffer(ctx, ತೂಕ, ನಿಜ),
                    ggml_vk_tensor_subbuffer(ctx, dst, ನಿಜ),
                    ggml_vk_subbuffer(ctx, ctx->prealloc_ಕೂಡಿಸು_rms_partials, ctx->prealloc_ಗಾತ್ರ_ಕೂಡಿಸು_rms_partials_ಸ್ಥಾನಾಂತರ),
                    ggml_vk_tensor_subbuffer(ctx, residual),
                    ggml_vk_tensor_subbuffer(ctx, post_scale),
                }, pc, { (uint32_t)CEIL_DIV(src0->ne[0], 128), 1, 1 });
        } ಇಲ್ಲದಿದ್ದರೆ {
            ggml_vk_dispatch_pipeline(ctx, subctx, pipeline,
                {
                    ggml_vk_tensor_subbuffer(ctx, src0, ನಿಜ),
                    ggml_vk_tensor_subbuffer(ctx, ತೂಕ, ನಿಜ),
                    ggml_vk_tensor_subbuffer(ctx, dst, ನಿಜ),
                    ggml_vk_tensor_subbuffer(ctx, residual),
                    ggml_vk_tensor_subbuffer(ctx, post_scale),
                }, pc, {
                    (uint32_t)src0->ne[1],
                    std::min((uint32_t)src0->ne[2], ctx->device->properties.limits.max_Compute_Work_ಗುಂಪು_ಎಣಿಕೆ[1]),
                    std::min((uint32_t)src0->ne[3], ctx->device->properties.limits.max_Compute_Work_ಗುಂಪು_ಎಣಿಕೆ[2]),
                });
        }
        ggml_vk_rms_norm_ಮುಕ್ತಾಯ(ctx, src0);
        ಹಿಂತಿರುಗಿ;
    }

    ggml_tensor * dst;
    ಸ್ಥಿರಾಂಕ ggml_tensor * src1;

    ಆದರೆ (ctx->fused_rms_norm_ಬಹುಲಕ != RMS_NORM_ಎಣಿಕೆ) {
        ggml_tensor * mul = cgraph->nodes[ಗಂಟು_idx + 1];
        dst = mul;
        src1 = mul->src[0] == rms ? mul->src[1] : mul->src[0];
    } ಇಲ್ಲದಿದ್ದರೆ {
        dst = rms;
        src1 = src0;
    }

    ಸ್ಥಿರಾಂಕ uint32_t ಅಂಕೆ_partials = ctx->do_ಕೂಡಿಸು_rms_partials ? ggml_vk_rms_ಅಂಕೆ_partials(ctx, dst) : 0;
    vk_op_ದ್ವಿಮಾನ_ತಳ್ಳು_constants bin = ggml_vk_rms_norm_ತಳ್ಳು_constants(src0, src1, dst, op_params[0], ಅಂಕೆ_partials);

    ಆದರೆ (ctx->fused_rms_norm_ಬಹುಲಕ == RMS_NORM_MUL_ROPE ||
        ctx->fused_rms_norm_ಬಹುಲಕ == RMS_NORM_MUL_ROPE_VIEW_ಗಣ_ಅಡ್ಡಸಾಲುಗಳು) {
        ಸ್ಥಿರ constexpr uint32_t max_tensors = 7;
        ಸ್ಥಿರಾಂಕ ggml_tensor *tensors[max_tensors] {};

        ggml_tensor *rms = cgraph->nodes[ಗಂಟು_idx + 0];
        ggml_tensor *mul = cgraph->nodes[ಗಂಟು_idx + 1];
        ggml_tensor *rope = cgraph->nodes[ಗಂಟು_idx + 2];

        ggml_tensor *other_src = mul->src[0] == rms ? mul->src[1] : mul->src[0];

        ತಾರ್ಕಿಕ do_ಗಣ_ಅಡ್ಡಸಾಲುಗಳು = ctx->fused_rms_norm_ಬಹುಲಕ == RMS_NORM_MUL_ROPE_VIEW_ಗಣ_ಅಡ್ಡಸಾಲುಗಳು;
        GGML_ASSERT(ctx->ಅಂಕೆ_additional_fused_ops == (do_ಗಣ_ಅಡ್ಡಸಾಲುಗಳು ? 4 : 2));

        tensors[0] = rms->src[0];
        tensors[1] = other_src;
        tensors[2] = mul;
        tensors[3] = rope->src[1]; // pos
        tensors[4] = rope->src[2]; // ff
        tensors[5] = cgraph->nodes[ಗಂಟು_idx + ctx->ಅಂಕೆ_additional_fused_ops]; // dst
        tensors[6] = do_ಗಣ_ಅಡ್ಡಸಾಲುಗಳು ? tensors[5]->src[1] : ಶೂನ್ಯಸೂಚಕ;
        ಸ್ಥಿರಾಂಕ uint32_t ಗಣ_ಅಡ್ಡಸಾಲುಗಳು_stride = do_ಗಣ_ಅಡ್ಡಸಾಲುಗಳು ? tensors[5]->nb[1] / ggml_type_ಗಾತ್ರ(tensors[5]->type) : 0;

        vk_op_rms_norm_mul_rope_ತಳ್ಳು_constants pc;
        pc.bin = bin;
        pc.rope = ggml_vk_make_rope_constants(rope, rope->src[0], tensors[4] != ಶೂನ್ಯಸೂಚಕ, ಸುಳ್ಳು, ಗಣ_ಅಡ್ಡಸಾಲುಗಳು_stride);

        vk_pipeline pipeline = tensors[5]->type == GGML_TYPE_F16 ? ctx->device->pipeline_rms_norm_mul_rope_f32_f16 : ctx->device->pipeline_rms_norm_mul_rope_f32_f32;

        ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);

        ggml_backend_vk_ಬಫರ್_context * buf_ctx[max_tensors];
        vk_ಬಫರ್ buf[max_tensors];
        size_t ಸ್ಥಾನಾಂತರ[max_tensors];
        ತಾರ್ಕಿಕ uma[max_tensors];

        ಗಾಗಿ (uint32_t i = 0; i < max_tensors; ++i) {
            ಆದರೆ (!tensors[i]) {
                // If any remaining descriptors are unused, just point them at src[0]
                buf[i] = buf[0];
                ಸ್ಥಾನಾಂತರ[i] = 0;
                ಮುಂದುವರಿ;
            }
            buf_ctx[i] = (ggml_backend_vk_ಬಫರ್_context *)tensors[i]->ಬಫರ್->context;
            buf[i] = ಶೂನ್ಯಸೂಚಕ;
            ಸ್ಥಾನಾಂತರ[i] = 0;
            uma[i] = ಸುಳ್ಳು;

            ಆದರೆ (ctx->device->uma) {
                ggml_vk_host_get(ctx->device, tensors[i]->ದತ್ತಾಂಶ, buf[i], ಸ್ಥಾನಾಂತರ[i]);
                uma[i] = buf[i] != ಶೂನ್ಯಸೂಚಕ;
            }
            ಆದರೆ (!uma[i]) {
                buf[i] = buf_ctx[i]->dev_ಬಫರ್;
                ಸ್ಥಾನಾಂತರ[i] = vk_tensor_ಸ್ಥಾನಾಂತರ(tensors[i]) + tensors[i]->view_offs;
            }
            GGML_ASSERT(buf[i] != ಶೂನ್ಯಸೂಚಕ);
        }

        // a_offset is unused (the fused path reads from shared memory), but the rope/set_rows dst can be misaligned.
        // Round the binding offset down to the storage buffer alignment; the in-element shift goes in pc.rope.d_offset.
        pc.rope.d_ಸ್ಥಾನಾಂತರ = get_misalign_ಬೈಟುಗಳು(ctx, tensors[5]) / ggml_type_ಗಾತ್ರ(tensors[5]->type);
        ಸ್ಥಾನಾಂತರ[5] &= ~(size_t(ctx->device->properties.limits.min_Storage_ಬಫರ್_ಸ್ಥಾನಾಂತರ_Alignment) - 1);

        std::array<uint32_t, 3> elements;
        elements = { (uint32_t)rms->src[0]->ne[1], (uint32_t)rms->src[0]->ne[2], (uint32_t)rms->src[0]->ne[3] };
        elements[1] = std::min(elements[1], ctx->device->properties.limits.max_Compute_Work_ಗುಂಪು_ಎಣಿಕೆ[1]);
        elements[2] = std::min(elements[2], ctx->device->properties.limits.max_Compute_Work_ಗುಂಪು_ಎಣಿಕೆ[2]);

        static_assert(max_tensors == 7);
        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline,
            {
                ggml_vk_subbuffer(ctx, buf[0], ಸ್ಥಾನಾಂತರ[0]),
                ggml_vk_subbuffer(ctx, buf[1], ಸ್ಥಾನಾಂತರ[1]),
                ggml_vk_subbuffer(ctx, buf[2], ಸ್ಥಾನಾಂತರ[2]),
                ggml_vk_subbuffer(ctx, buf[3], ಸ್ಥಾನಾಂತರ[3]),
                ggml_vk_subbuffer(ctx, buf[4], ಸ್ಥಾನಾಂತರ[4]),
                ggml_vk_subbuffer(ctx, buf[5], ಸ್ಥಾನಾಂತರ[5]),
                ggml_vk_subbuffer(ctx, buf[6], ಸ್ಥಾನಾಂತರ[6]),
            }, pc, elements);
    } ಇಲ್ಲದಿದ್ದರೆ {
        GGML_ASSERT(ctx->fused_rms_norm_ಬಹುಲಕ == RMS_NORM_MUL || ctx->fused_rms_norm_ಬಹುಲಕ == RMS_NORM_ಎಣಿಕೆ);
        ggml_vk_op_f32<vk_op_ದ್ವಿಮಾನ_ತಳ್ಳು_constants>(ctx, subctx, src0, src1, ಶೂನ್ಯಸೂಚಕ, ಶೂನ್ಯಸೂಚಕ, dst, GGML_OP_RMS_NORM, std::move(bin));
    }

    ggml_vk_rms_norm_ಮುಕ್ತಾಯ(ctx, src0);
}

ಶೂನ್ಯ ggml_vk_rms_norm_ಹಿಂಭಾಗ(ggml_backend_vk_context * ctx, vk_context& subctx, ಸ್ಥಿರಾಂಕ ggml_tensor * src0, ಸ್ಥಿರಾಂಕ ggml_tensor * src1, ggml_tensor * dst) {
    ದಶಮಾಂಶ * op_params = (ದಶಮಾಂಶ *)dst->op_params;
    ggml_vk_op_f32<vk_op_ತಳ್ಳು_constants>(ctx, subctx, src0, src1, ಶೂನ್ಯಸೂಚಕ, ಶೂನ್ಯಸೂಚಕ, dst, GGML_OP_RMS_NORM_ಹಿಂಭಾಗ, { (uint32_t)src0->ne[0], (uint32_t)src0->ne[1], op_params[0], 0.0f, 0.0f, 0.0f });
}

ಶೂನ್ಯ ggml_vk_l2_norm(ggml_backend_vk_context * ctx, vk_context& subctx, ಸ್ಥಿರಾಂಕ ggml_tensor * src0, ggml_tensor * dst) {
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * op_params = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *)dst->op_params;
    vk_op_unary_ತಳ್ಳು_constants p = vk_op_unary_ತಳ್ಳು_constants_init(src0, dst);
    p.param1 = op_params[0];
    ggml_vk_op_f32<vk_op_unary_ತಳ್ಳು_constants>(ctx, subctx, src0, ಶೂನ್ಯಸೂಚಕ, ಶೂನ್ಯಸೂಚಕ, ಶೂನ್ಯಸೂಚಕ, dst, GGML_OP_L2_NORM, std::move(p));
}

ಶೂನ್ಯ ggml_vk_unary(ggml_backend_vk_context * ctx, vk_context& subctx, ಸ್ಥಿರಾಂಕ ggml_tensor * src0, ggml_tensor * dst) {
    ggml_vk_op_f32(ctx, subctx, src0, ಶೂನ್ಯಸೂಚಕ, ಶೂನ್ಯಸೂಚಕ, ಶೂನ್ಯಸೂಚಕ, dst, GGML_OP_UNARY, vk_op_unary_ತಳ್ಳು_constants_init(src0, dst));
}

ಶೂನ್ಯ ggml_vk_xielu(ggml_backend_vk_context * ctx, vk_context& subctx, ಸ್ಥಿರಾಂಕ ggml_tensor * src0, ggml_tensor * dst) {
    ದಶಮಾಂಶ * op_params = (ದಶಮಾಂಶ *)dst->op_params;
    vk_op_unary_ತಳ್ಳು_constants p = vk_op_unary_ತಳ್ಳು_constants_init(src0, dst);
    p.param1 = op_params[1];
    p.param2 = op_params[2];
    p.param3 = op_params[3];
    p.param4 = op_params[4];
    ggml_vk_op_f32(ctx, subctx, src0, ಶೂನ್ಯಸೂಚಕ, ಶೂನ್ಯಸೂಚಕ, ಶೂನ್ಯಸೂಚಕ, dst, GGML_OP_UNARY, std::move(p));
}

ಶೂನ್ಯ ggml_vk_glu(ggml_backend_vk_context * ctx, vk_context& subctx, ಸ್ಥಿರಾಂಕ ggml_tensor * src0, ಸ್ಥಿರಾಂಕ ggml_tensor * src1, ggml_tensor * dst) {
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * op_params_f = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *)dst->op_params;

    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ swapped = (ತಾರ್ಕಿಕ)dst->op_params[1];
    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ ವಿಭಜಿಸು = src1 != ಶೂನ್ಯಸೂಚಕ;
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ alpha = op_params_f[2];
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ ಎಲ್ಲೆ = op_params_f[3];

    ಆದರೆ (!ವಿಭಜಿಸು) {
        GGML_ASSERT(src0->ne[0] / 2 == dst->ne[0]);
    } ಇಲ್ಲದಿದ್ದರೆ {
        GGML_ASSERT(src0->ne[0] == src1->ne[0]);
        GGML_ASSERT(src0->ne[0] == dst->ne[0]);
        GGML_ASSERT(src0->type == src1->type);
    }

    ಸ್ಥಿರಾಂಕ uint32_t mode = ವಿಭಜಿಸು ? 2 : (swapped ? 1 : 0);
    ಸ್ಥಿರಾಂಕ uint32_t src0_type_ಗಾತ್ರ = ggml_type_ಗಾತ್ರ(src0->type);
    ಸ್ಥಿರಾಂಕ uint32_t src1_type_ಗಾತ್ರ = ವಿಭಜಿಸು ? ggml_type_ಗಾತ್ರ(src1->type) : src0_type_ಗಾತ್ರ;
    ಸ್ಥಿರಾಂಕ uint32_t dst_type_ಗಾತ್ರ  = ggml_type_ಗಾತ್ರ(dst->type);

    ggml_vk_op_f32<vk_op_glu_ತಳ್ಳು_constants>(ctx, subctx, src0, src1, ಶೂನ್ಯಸೂಚಕ, ಶೂನ್ಯಸೂಚಕ, dst, GGML_OP_GLU,
        {
            (uint32_t)ggml_nelements(dst),
            (uint32_t)src0->ne[0],
            (uint32_t)dst->ne[0],
            mode,
            alpha,
            ಎಲ್ಲೆ,
            (uint32_t)(src0->nb[0] / src0_type_ಗಾತ್ರ),
            (uint32_t)(src0->nb[1] / src0_type_ಗಾತ್ರ),
            (uint32_t)(src0->nb[2] / src0_type_ಗಾತ್ರ),
            (uint32_t)(src0->nb[3] / src0_type_ಗಾತ್ರ),
            (uint32_t)((ವಿಭಜಿಸು ? src1->nb[0] : src0->nb[0]) / src1_type_ಗಾತ್ರ),
            (uint32_t)((ವಿಭಜಿಸು ? src1->nb[1] : src0->nb[1]) / src1_type_ಗಾತ್ರ),
            (uint32_t)((ವಿಭಜಿಸು ? src1->nb[2] : src0->nb[2]) / src1_type_ಗಾತ್ರ),
            (uint32_t)((ವಿಭಜಿಸು ? src1->nb[3] : src0->nb[3]) / src1_type_ಗಾತ್ರ),
            (uint32_t)(dst->nb[0] / dst_type_ಗಾತ್ರ),
            (uint32_t)(dst->nb[1] / dst_type_ಗಾತ್ರ),
            (uint32_t)(dst->nb[2] / dst_type_ಗಾತ್ರ),
            (uint32_t)(dst->nb[3] / dst_type_ಗಾತ್ರ),
            (uint32_t)dst->ne[1],
            (uint32_t)dst->ne[2],
            0,
            0, 0, 0, 0, 0, 0,
        });
}

ಶೂನ್ಯ ggml_vk_diag_ಮುಖವಾಡ_inf(ggml_backend_vk_context * ctx, vk_context& subctx, ಸ್ಥಿರಾಂಕ ggml_tensor * src0, ggml_tensor * dst) {
    int32_t * op_params = (int32_t *)dst->op_params;
    ggml_vk_op_f32<vk_op_diag_ಮುಖವಾಡ_ತಳ್ಳು_constants>(ctx, subctx, src0, ಶೂನ್ಯಸೂಚಕ, ಶೂನ್ಯಸೂಚಕ, ಶೂನ್ಯಸೂಚಕ, dst, GGML_OP_DIAG_ಮುಖವಾಡ_INF, { (uint32_t)src0->ne[0], (uint32_t)src0->ne[1], op_params[0] });
}

ಶೂನ್ಯ ggml_vk_soft_max(ggml_backend_vk_context * ctx, vk_context& subctx, ಸ್ಥಿರಾಂಕ ggml_tensor * src0, ಸ್ಥಿರಾಂಕ ggml_tensor * src1, ಸ್ಥಿರಾಂಕ ggml_tensor * src2, ggml_tensor * dst) {
    ದಶಮಾಂಶ * op_params = (ದಶಮಾಂಶ *)dst->op_params;

    ದಶಮಾಂಶ scale = op_params[0];
    ದಶಮಾಂಶ max_bias = op_params[1];

    ಸ್ಥಿರಾಂಕ uint32_t ncols =   (uint32_t)src0->ne[0];
    ಸ್ಥಿರಾಂಕ uint32_t nrows_x = (uint32_t)ggml_nrows(src0);
    ಸ್ಥಿರಾಂಕ uint32_t nrows_y = (uint32_t)src0->ne[1];

    ಸ್ಥಿರಾಂಕ uint32_t ne12 = src1 ? (uint32_t)(src1->ne[2]) : 0u;
    ಸ್ಥಿರಾಂಕ uint32_t ne13 = src1 ? (uint32_t)(src1->ne[3]) : 0u;
    ಸ್ಥಿರಾಂಕ uint32_t nb11 = src1 ? (uint32_t)(src1->nb[1] / src1->nb[0]) : 0u;
    ಸ್ಥಿರಾಂಕ uint32_t nb12 = src1 ? (uint32_t)(src1->nb[2] / src1->nb[0]) : 0u;
    ಸ್ಥಿರಾಂಕ uint32_t nb13 = src1 ? (uint32_t)(src1->nb[3] / src1->nb[0]) : 0u;

    ಸ್ಥಿರಾಂಕ uint32_t n_ತಲೆ_kv   = src0->ne[2];
    ಸ್ಥಿರಾಂಕ uint32_t n_ತಲೆ_log2 = 1u << (uint32_t) floorf(log2f((ದಶಮಾಂಶ) n_ತಲೆ_kv));

    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ m0 = powf(2.0f, -(max_bias       ) / n_ತಲೆ_log2);
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ m1 = powf(2.0f, -(max_bias / 2.0f) / n_ತಲೆ_log2);

    vk_op_soft_max_ತಳ್ಳು_constants pc {
        ncols,
        src1 != ಶೂನ್ಯಸೂಚಕ ? nrows_y : (uint32_t)0,
        (uint32_t)src0->ne[0], (uint32_t)src0->ne[1], (uint32_t)src0->ne[2],
        ne12, ne13,
        nb11, nb12, nb13,
        scale, max_bias,
        m0, m1,
        n_ತಲೆ_log2,
        nrows_x,
        src2 != ಶೂನ್ಯಸೂಚಕ
    };

    ಆದರೆ (ncols <= 16384) {
        ggml_vk_op_f32<vk_op_soft_max_ತಳ್ಳು_constants>(ctx, subctx, src0, src1, src2, ಶೂನ್ಯಸೂಚಕ, dst, GGML_OP_SOFT_MAX, std::move(pc));
    } ಇಲ್ಲದಿದ್ದರೆ {

        vk_subbuffer buf_a = ggml_vk_tensor_subbuffer(ctx, src0);
        vk_subbuffer buf_b = src1 ? ggml_vk_tensor_subbuffer(ctx, src1) : buf_a;
        vk_subbuffer buf_c = src2 ? ggml_vk_tensor_subbuffer(ctx, src2) : buf_a;
        vk_subbuffer buf_d = ggml_vk_tensor_subbuffer(ctx, dst);

        uint32_t elems_ಪ್ರತಿ_wg = 128 * 4;
        uint32_t ಅಂಕೆ_wgs = CEIL_DIV(ncols, elems_ಪ್ರತಿ_wg);
        size_t tmp_ಗಾತ್ರ = ಅಂಕೆ_wgs * nrows_x * ಗಾತ್ರ(ದಶಮಾಂಶ);

        ಆದರೆ (ctx->prealloc_ಗಾತ್ರ_x < tmp_ಗಾತ್ರ) {
            ctx->prealloc_ಗಾತ್ರ_x = tmp_ಗಾತ್ರ;
            ggml_vk_preallocate_buffers(ctx, subctx);
        }
        ಆದರೆ (ctx->prealloc_ಗಾತ್ರ_y < tmp_ಗಾತ್ರ) {
            ctx->prealloc_ಗಾತ್ರ_y = tmp_ಗಾತ್ರ;
            ggml_vk_preallocate_buffers(ctx, subctx);
        }
        ಆದರೆ (ctx->prealloc_x_need_sync || ctx->prealloc_y_need_sync) {
            ggml_vk_sync_buffers(ctx, subctx);
        }

        vk_subbuffer buf_x = { ctx->prealloc_x, 0, tmp_ಗಾತ್ರ };
        vk_subbuffer buf_y = { ctx->prealloc_y, 0, tmp_ಗಾತ್ರ };

        // the partial results overwrite a matmul input converted into prealloc_y
        ctx->prealloc_y_ಕೊನೆಯ_pipeline_ಬಳಸಿದ = ಶೂನ್ಯಸೂಚಕ;
        ctx->prealloc_y_ಕೊನೆಯ_tensor_ಬಳಸಿದ = ಶೂನ್ಯಸೂಚಕ;

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

        ctx->prealloc_x_need_sync = ನಿಜ;
        ctx->prealloc_y_need_sync = ನಿಜ;
    }
}

ಶೂನ್ಯ ggml_vk_soft_max_ಹಿಂಭಾಗ(ggml_backend_vk_context * ctx, vk_context& subctx, ಸ್ಥಿರಾಂಕ ggml_tensor * src0, ಸ್ಥಿರಾಂಕ ggml_tensor * src1, ggml_tensor * dst) {
    ದಶಮಾಂಶ * op_params = (ದಶಮಾಂಶ *)dst->op_params;
    ggml_vk_op_f32<vk_op_ತಳ್ಳು_constants>(ctx, subctx, src0, src1, ಶೂನ್ಯಸೂಚಕ, ಶೂನ್ಯಸೂಚಕ, dst, GGML_OP_SOFT_MAX_ಹಿಂಭಾಗ, { (uint32_t)src0->ne[0], (uint32_t)ggml_nrows(src0), op_params[0], op_params[1], 0.0f, 0.0f });
}

ಶೂನ್ಯ ggml_vk_topk_moe(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_cgraph * cgraph, ಪೂರ್ಣಾಂಕ ಗಂಟು_idx) {
    topk_moe_ಬಹುಲಕ mode = ctx->fused_topk_moe_ಬಹುಲಕ;
    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ has_bias = mode == TOPK_MOE_SIGMOID_NORM_BIAS || mode == TOPK_MOE_SQRT_SOFTPLUS_NORM_BIAS;
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

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ n_experts = logits->ne[0];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ n_ಅಡ್ಡಸಾಲುಗಳು    = logits->ne[1];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ n_expert_ಬಳಸಿದ = weights->ne[1];

    GGML_ASSERT(ids->nb[1] / ggml_type_ಗಾತ್ರ(ids->type) == (size_t) n_experts);

    vk_pipeline pipeline = ggml_vk_op_get_pipeline(ctx, ಶೂನ್ಯಸೂಚಕ, ಶೂನ್ಯಸೂಚಕ, ಶೂನ್ಯಸೂಚಕ, cgraph->nodes[ಗಂಟು_idx], GGML_OP_SOFT_MAX);

    ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);

    vk_subbuffer logits_buf = ggml_vk_tensor_subbuffer(ctx, logits);
    vk_subbuffer bias_buf = ggml_vk_tensor_subbuffer(ctx, bias);
    vk_subbuffer weights_buf = ggml_vk_tensor_subbuffer(ctx, weights);
    vk_subbuffer ids_buf = ggml_vk_tensor_subbuffer(ctx, ids);

    vk_op_topk_moe_ತಳ್ಳು_constants pc {};
    pc.n_ಅಡ್ಡಸಾಲುಗಳು = n_ಅಡ್ಡಸಾಲುಗಳು;
    pc.n_experts_ತಳ್ಳು = n_experts;
    pc.n_expert_ಬಳಸಿದ = n_expert_ಬಳಸಿದ;
    pc.clamp_min = -std::numeric_limits<ದಶಮಾಂಶ>::infinity();
    pc.clamp_max = std::numeric_limits<ದಶಮಾಂಶ>::infinity();
    ಆದರೆ (mode == TOPK_MOE_EARLY_SOFTMAX_NORM) {
        ggml_tensor * clamp = cgraph->nodes[ಗಂಟು_idx + 7];
        GGML_ASSERT(clamp->op == GGML_OP_CLAMP);
        pc.clamp_min = ggml_get_op_params_f32(clamp, 0);
        pc.clamp_max = ggml_get_op_params_f32(clamp, 1);
    }
    ಆದರೆ (mode == TOPK_MOE_SIGMOID_NORM_BIAS) {
        ggml_tensor * clamp = cgraph->nodes[ಗಂಟು_idx + 8];
        GGML_ASSERT(clamp->op == GGML_OP_CLAMP);
        pc.clamp_min = ggml_get_op_params_f32(clamp, 0);
        pc.clamp_max = ggml_get_op_params_f32(clamp, 1);
    }
    ಆದರೆ (mode == TOPK_MOE_SQRT_SOFTPLUS_NORM_BIAS) {
        ggml_tensor * clamp = cgraph->nodes[ಗಂಟು_idx + 9];
        GGML_ASSERT(clamp->op == GGML_OP_CLAMP);
        pc.clamp_min = ggml_get_op_params_f32(clamp, 0);
        pc.clamp_max = ggml_get_op_params_f32(clamp, 1);
    }

#ವ್ಯಾಖ್ಯಾನಿಸು GATING_FUNC_SOFTMAX 0
#ವ್ಯಾಖ್ಯಾನಿಸು GATING_FUNC_SIGMOID 1
#ವ್ಯಾಖ್ಯಾನಿಸು GATING_FUNC_SOFTMAX_ತೂಕ 2
#ವ್ಯಾಖ್ಯಾನಿಸು GATING_FUNC_SQRT_SOFTPLUS 3

    pc.gating_func = mode == TOPK_MOE_SIGMOID_NORM_BIAS       ? GATING_FUNC_SIGMOID :
                     mode == TOPK_MOE_SQRT_SOFTPLUS_NORM_BIAS ? GATING_FUNC_SQRT_SOFTPLUS :
                     mode == TOPK_MOE_LATE_SOFTMAX             ? GATING_FUNC_SOFTMAX_ತೂಕ :
                                                                 GATING_FUNC_SOFTMAX;
    pc.has_bias = has_bias;
    pc.with_norm = mode == TOPK_MOE_EARLY_SOFTMAX_NORM || has_bias;
    ಆದರೆ (ctx->fused_topk_moe_scale) {
        GGML_ASSERT(weights->op == GGML_OP_SCALE);
        pc.output_scale = ggml_get_op_params_f32(weights, 0);
        pc.output_bias = ggml_get_op_params_f32(weights, 1);
    } ಇಲ್ಲದಿದ್ದರೆ {
        pc.output_scale = 1.0f;
        pc.output_bias = 0.0f;
    }

    GGML_ASSERT(n_expert_ಬಳಸಿದ <= n_experts);

    ಸ್ಥಿರಾಂಕ uint32_t ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_ಖಂಡ = 4;
    std::array<uint32_t, 3> elements = { CEIL_DIV(n_ಅಡ್ಡಸಾಲುಗಳು, ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_ಖಂಡ), 1, 1 };

    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, {logits_buf, bias_buf, weights_buf, ids_buf}, pc, elements);
}

ಶೂನ್ಯ ggml_vk_rope(ggml_backend_vk_context * ctx, vk_context& subctx, ಸ್ಥಿರಾಂಕ ggml_cgraph * cgraph, ಪೂರ್ಣಾಂಕ ಗಂಟು_idx, ತಾರ್ಕಿಕ backprop) {
    ggml_tensor * dst = cgraph->nodes[ಗಂಟು_idx];
    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = dst->src[0];
    ಸ್ಥಿರಾಂಕ ggml_tensor * src1 = dst->src[1];
    ಸ್ಥಿರಾಂಕ ggml_tensor * src2 = dst->src[2];
    ಸ್ಥಿರಾಂಕ ggml_tensor * src3 = ಶೂನ್ಯಸೂಚಕ;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ n_dims        = ((int32_t *) dst->op_params)[1];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ mode          = ((int32_t *) dst->op_params)[2];
    // const int n_ctx         = ((int32_t *) dst->op_params)[3];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ n_ctx_orig    = ((int32_t *) dst->op_params)[4];
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ freq_ತಳ   = ((ದಶಮಾಂಶ *)   dst->op_params)[5];
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ beta_fast   = ((ದಶಮಾಂಶ *)   dst->op_params)[9];
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ beta_slow   = ((ದಶಮಾಂಶ *)   dst->op_params)[10];
    ಪೂರ್ಣಾಂಕ sections[4] {};
    ಆದರೆ (mode & GGML_ROPE_TYPE_MROPE) {
        ಸ್ಮರಣೆ_ನಕಲು(sections, (int32_t *) dst->op_params + 11, ಗಾತ್ರ(ಪೂರ್ಣಾಂಕ)*4);
    }

    ದಶಮಾಂಶ corr_dims[2];
    ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_ತಳ, beta_fast, beta_slow, corr_dims);

    uint32_t ಗಣ_ಅಡ್ಡಸಾಲುಗಳು_stride = 0;
    // Fused rope + view + set_rows passes the set_rows destination stride in set_rows_stride
    // and overrides the dst and sets src3=row_indices
    ಆದರೆ (ctx->ಅಂಕೆ_additional_fused_ops > 0) {
        ಗಣ_ಅಡ್ಡಸಾಲುಗಳು_stride = cgraph->nodes[ಗಂಟು_idx + 2]->nb[1] / ggml_type_ಗಾತ್ರ(cgraph->nodes[ಗಂಟು_idx + 2]->type);
        src3 = cgraph->nodes[ಗಂಟು_idx + 2]->src[1];
        dst = cgraph->nodes[ಗಂಟು_idx + 2];
    }

    ggml_vk_op_f32<vk_op_rope_ತಳ್ಳು_constants>(ctx, subctx, src0, src1, src2, src3, dst, GGML_OP_ROPE,
        ggml_vk_make_rope_constants(cgraph->nodes[ಗಂಟು_idx], src0, src2 != ಶೂನ್ಯಸೂಚಕ, backprop, ಗಣ_ಅಡ್ಡಸಾಲುಗಳು_stride));
}

ಶೂನ್ಯ ggml_vk_argsort(ggml_backend_vk_context * ctx, vk_context& subctx, ಸ್ಥಿರಾಂಕ ggml_tensor * src0, ggml_tensor * dst) {
    ಸ್ಥಿರಾಂಕ uint32_t * op_params = (ಸ್ಥಿರಾಂಕ uint32_t *)dst->op_params;

    uint32_t ncols = src0->ne[0];
    uint32_t nrows = ggml_nrows(src0);

    uint32_t ncols_pad_log2 = (uint32_t)ceilf(log2f(ದಶಮಾಂಶ(ncols)));
    uint32_t ncolsp2 = 1 << ncols_pad_log2;

    vk_op_argsort_ತಳ್ಳು_constants pc { ncols, ncolsp2, ncols_pad_log2, nrows, op_params[0], 0, 0, 0, 0, };

    // Pick the largest workgroup size <= ncolsp2
    uint32_t pipeline_idx = std::min(ncols_pad_log2, ಅಂಕೆ_argsort_pipelines - 1);

    uint32_t max_wg_log2 = std::min(ctx->device->max_workgroup_ಗಾತ್ರ_log2, ಅಂಕೆ_argsort_pipelines - 1);

    // Use the "small" argsort shader if the whole sort can be done by a single workgroup.
    ತಾರ್ಕಿಕ use_small = ncols_pad_log2 <= max_wg_log2 &&
                     ctx->device->pipeline_argsort_f32[pipeline_idx] != ಶೂನ್ಯಸೂಚಕ;

    vk_pipeline pipeline = use_small ? ctx->device->pipeline_argsort_f32[pipeline_idx]
                                     : ctx->device->pipeline_argsort_large_f32[pipeline_idx];

    vk_subbuffer src0_buf = ggml_vk_tensor_subbuffer(ctx, src0);
    vk_subbuffer dst_buf = ggml_vk_tensor_subbuffer(ctx, dst);
    vk_subbuffer subbuf1 = dst_buf;

    // Reserve space for ivec2 per element, with rows padded to a power of two
    ಆದರೆ (!use_small) {
        ಸ್ಥಿರಾಂಕ size_t x_sz = size_t{ncolsp2} * nrows * 2 * ಗಾತ್ರ(ಪೂರ್ಣಾಂಕ);

        ಆದರೆ (ctx->prealloc_ಗಾತ್ರ_x < x_sz) {
            ctx->prealloc_ಗಾತ್ರ_x = x_sz;
            ggml_vk_preallocate_buffers(ctx, subctx);
        }
        ಆದರೆ (ctx->prealloc_x_need_sync) {
            ggml_vk_sync_buffers(ctx, subctx);
        }
        subbuf1 = { ctx->prealloc_x, 0, ctx->prealloc_x->size };
    }

    std::array<uint32_t, 3> elements;

    elements[0] = ncolsp2;
    elements[1] = std::min((uint32_t)ggml_nrows(src0), ctx->device->properties.limits.max_Compute_Work_ಗುಂಪು_ಎಣಿಕೆ[1]);
    elements[2] = 1;

    // First dispatch initializes tmp_idx and does the first N passes where
    // there is only communication between threads in the same workgroup.
    {
        vk_op_argsort_ತಳ್ಳು_constants pc2 = pc;
        pc2.outer_ಆರಂಭ = 0;
        pc2.outer_ಕೊನೆ = std::min(ncols_pad_log2, max_wg_log2);
        pc2.inner_ಆರಂಭ = 0;
        pc2.inner_ಕೊನೆ = 100;
        ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);
        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { src0_buf, subbuf1, dst_buf }, pc2, elements);
    }
    ಆದರೆ (!use_small) {
        ggml_vk_sync_buffers(ctx, subctx);
        // Loop over outer/inner passes, synchronizing between each pass.
        ಗಾಗಿ (uint32_t outer = max_wg_log2; outer < ncols_pad_log2; ++outer) {
            ಗಾಗಿ (uint32_t inner = 0; inner < outer + 1; ++inner) {
                vk_op_argsort_ತಳ್ಳು_constants pc2 = pc;
                pc2.outer_ಆರಂಭ = outer;
                pc2.outer_ಕೊನೆ = outer + 1;
                pc2.inner_ಆರಂಭ = inner;
                pc2.inner_ಕೊನೆ = inner + 1;
                // When the inner idx is large enough, there's only communication
                // within a workgroup. So the remaining inner iterations can all
                // run in the same dispatch.
                ಆದರೆ (outer - inner < pipeline_idx) {
                    pc2.inner_ಕೊನೆ = 100;
                    inner = outer;
                    pipeline = ctx->device->pipeline_argsort_large_f32[pipeline_idx];
                } ಇಲ್ಲದಿದ್ದರೆ {
                    // Smaller workgroup empirically seems to perform better
                    pipeline = ctx->device->pipeline_argsort_large_f32[pipeline_idx - 2];
                }
                ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);
                ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { src0_buf, subbuf1, dst_buf }, pc2, elements);
                ggml_vk_sync_buffers(ctx, subctx);
            }
        }
        ctx->prealloc_x_need_sync = ನಿಜ;
    }
}

ಶೂನ್ಯ ggml_vk_topk(ggml_backend_vk_context * ctx, vk_context& subctx, ಸ್ಥಿರಾಂಕ ggml_tensor * src0, ggml_tensor * dst) {
    uint32_t ncols = src0->ne[0];
    uint32_t nrows = ggml_nrows(src0);
    uint32_t k = dst->ne[0];

    // tournament path is faster where it fits; use radix-select only past its k limit
    ಸ್ಥಿರಾಂಕ uint32_t k_min_pipeline = std::max((uint32_t) log2f(ದಶಮಾಂಶ(k)) + 1, ctx->device->subgroup_ಗಾತ್ರ_log2);
    ಆದರೆ (k_min_pipeline >= ಅಂಕೆ_topk_pipelines || ctx->device->pipeline_topk_f32[k_min_pipeline] == ಶೂನ್ಯಸೂಚಕ) {
        vk_pipeline pipeline = ctx->device->pipeline_topk_radix_f32;
        GGML_ASSERT(pipeline != ಶೂನ್ಯಸೂಚಕ);

        ಆದರೆ (ctx->prealloc_x_need_sync) {
            ggml_vk_sync_buffers(ctx, subctx);
        }

        vk_op_topk_radix_ತಳ್ಳು_constants pc { ncols, k, nrows, 0, 0, 0 };
        std::array<uint32_t, 3> elements {
            pipeline->wg_denoms[0],
            std::min(nrows, ctx->device->properties.limits.max_Compute_Work_ಗುಂಪು_ಎಣಿಕೆ[1]),
            1,
        };
        // the non-QSA path only uses bindings 0/1; bind valid buffers for the unused QSA slots
        vk_subbuffer src0_buf = ggml_vk_tensor_subbuffer(ctx, src0);
        vk_subbuffer dst_buf  = ggml_vk_tensor_subbuffer(ctx, dst);
        ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);
        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline,
            { src0_buf, dst_buf, src0_buf, src0_buf, src0_buf }, pc, elements);
        ಹಿಂತಿರುಗಿ;
    }

    vk_op_topk_ತಳ್ಳು_constants pc { ncols, ncols, ncols, k, nrows, 0, 0 };

    ಆದರೆ (ctx->prealloc_x_need_sync) {
        ggml_vk_sync_buffers(ctx, subctx);
    }

    std::array<uint32_t, 3> elements;
    elements[1] = std::min(nrows, ctx->device->properties.limits.max_Compute_Work_ಗುಂಪು_ಎಣಿಕೆ[1]);
    elements[2] = 1;

    uint32_t ಅಂಕೆ_elements = ncols;

    // Each iteration reduces a workgroup's worth of elements down to the K
    // largest elements. Repeat until we have the top K elements.
    // Need to do at least one iteration to write out the results.
    ತಾರ್ಕಿಕ ಮುಗಿದ_one_iter = ಸುಳ್ಳು;
    uint32_t dbl_buf_ಸೂಚ್ಯಂಕ = 0;
    size_t dbl_buf_ಗಾತ್ರ;
    ಆಗ (ಅಂಕೆ_elements > k || !ಮುಗಿದ_one_iter) {

        // Prefer going as small as num_topk_pipelines - 3 for perf reasons.
        // But if K is larger, then we need a larger workgroup
        uint32_t max_pipeline = ಅಂಕೆ_topk_pipelines - 1;
        uint32_t preferred_pipeline = std::max(ಅಂಕೆ_topk_pipelines - 3, (uint32_t)log2f(ದಶಮಾಂಶ(k)) + 2);
        max_pipeline = std::min(preferred_pipeline, max_pipeline);
        uint32_t min_pipeline = (uint32_t)log2f(ದಶಮಾಂಶ(k)) + 1;
        // require full subgroup
        min_pipeline = std::max(min_pipeline, ctx->device->subgroup_ಗಾತ್ರ_log2);

        uint32_t pipeline_idx = (uint32_t)ceilf(log2f(ದಶಮಾಂಶ(ಅಂಕೆ_elements)));
        pipeline_idx = std::min(pipeline_idx, max_pipeline);
        pipeline_idx = std::max(pipeline_idx, min_pipeline);

        ಆದರೆ (ಅಂಕೆ_elements > (1u << pipeline_idx)) {
            // If we could finish on this loop iteration (i.e. a single workgroup)
            // then do so. It's better than the overhead of another pass.
            ಗಾಗಿ (uint32_t i = pipeline_idx; i < ಅಂಕೆ_topk_pipelines; ++i) {
                ಆದರೆ (ಅಂಕೆ_elements <= (1u << i)) {
                    pipeline_idx = i;
                    ನಿಲ್ಲಿಸು;
                }
            }
        }

        vk_pipeline pipeline = ctx->device->pipeline_topk_f32[pipeline_idx];
        // If the device doesn't support a pipeline this large, use smaller
        ಆಗ (!pipeline) {
            pipeline_idx--;
            GGML_ASSERT(pipeline_idx >= min_pipeline);
            pipeline = ctx->device->pipeline_topk_f32[pipeline_idx];
        }

        vk_op_topk_ತಳ್ಳು_constants pc2 = pc;
        pc2.ncols_ಒಳಹರಿವು = ಅಂಕೆ_elements;

        // Number of elements remaining after this pass
        uint32_t ಅಂಕೆ_dst_elements = (ಅಂಕೆ_elements / pipeline->wg_denoms[0]) * k + std::min(k, ಅಂಕೆ_elements % pipeline->wg_denoms[0]);

        pc2.ncols_output = ಅಂಕೆ_dst_elements;

        ಆದರೆ (!ಮುಗಿದ_one_iter) {
            // Reserve space for ivec2 per element, double buffered
            // K per workgroup per row
            dbl_buf_ಗಾತ್ರ = ಅಂಕೆ_dst_elements * nrows * 2 * ಗಾತ್ರ(ಪೂರ್ಣಾಂಕ);
            dbl_buf_ಗಾತ್ರ = ROUNDUP_POW2(dbl_buf_ಗಾತ್ರ, ctx->device->properties.limits.min_Storage_ಬಫರ್_ಸ್ಥಾನಾಂತರ_Alignment);
            ಸ್ಥಿರಾಂಕ size_t x_sz = dbl_buf_ಗಾತ್ರ * 2;

            ಆದರೆ (ctx->prealloc_ಗಾತ್ರ_x < x_sz) {
                ctx->prealloc_ಗಾತ್ರ_x = x_sz;
                ggml_vk_preallocate_buffers(ctx, subctx);
            }
        }

        vk_subbuffer src_buf;
        vk_subbuffer dst_buf;

        ಆದರೆ (ಅಂಕೆ_elements == ncols) {
            pc2.ಮೊದಲ_pass = 1;
            src_buf = ggml_vk_tensor_subbuffer(ctx, src0);
        } ಇಲ್ಲದಿದ್ದರೆ {
            src_buf = { ctx->prealloc_x, dbl_buf_ಸೂಚ್ಯಂಕ * dbl_buf_ಗಾತ್ರ, dbl_buf_ಗಾತ್ರ };
        }
        ಆದರೆ (ಅಂಕೆ_dst_elements == k) {
            pc2.ಕೊನೆಯ_pass = 1;
            dst_buf = ggml_vk_tensor_subbuffer(ctx, dst);
        } ಇಲ್ಲದಿದ್ದರೆ {
            dst_buf = { ctx->prealloc_x, (dbl_buf_ಸೂಚ್ಯಂಕ ^ 1) * dbl_buf_ಗಾತ್ರ, dbl_buf_ಗಾತ್ರ };
        }

        elements[0] = ಅಂಕೆ_elements;

        ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);
        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { src_buf, dst_buf }, pc2, elements);
        ಅಂಕೆ_elements = ಅಂಕೆ_dst_elements;
        dbl_buf_ಸೂಚ್ಯಂಕ ^= 1;
        ಆದರೆ (ಅಂಕೆ_elements > k) {
            ggml_vk_sync_buffers(ctx, subctx);
        }
        ಮುಗಿದ_one_iter = ನಿಜ;
    }
    ctx->prealloc_x_need_sync = ನಿಜ;
}

ಶೂನ್ಯ ggml_vk_topk_qsa(ggml_backend_vk_context * ctx, vk_context& subctx, ಸ್ಥಿರಾಂಕ ggml_cgraph * cgraph, ಪೂರ್ಣಾಂಕ ಗಂಟು_idx) {
    ಸ್ಥಿರಾಂಕ ggml_tensor * get_ಅಡ್ಡಸಾಲುಗಳು = cgraph->nodes[ಗಂಟು_idx + 0];
    ಸ್ಥಿರಾಂಕ ggml_tensor * ಕೂಡಿಸು_2      = cgraph->nodes[ಗಂಟು_idx + ctx->ಅಂಕೆ_additional_fused_ops - 1];
    ggml_tensor *       ತುದಿ_k    = cgraph->nodes[ಗಂಟು_idx + ctx->ಅಂಕೆ_additional_fused_ops];

    ಸ್ಥಿರಾಂಕ ggml_tensor * scores   = get_ಅಡ್ಡಸಾಲುಗಳು->src[0]; // [n_tps, n_blocks, n_stream]
    ಸ್ಥಿರಾಂಕ ggml_tensor * cell_blk = get_ಅಡ್ಡಸಾಲುಗಳು->src[1]; // [n_kv, n_stream]

    // raw f16 mask: follow the reshape/cpy chain back to the materialized input
    ಸ್ಥಿರಾಂಕ ggml_tensor * ಮುಖವಾಡ = ಕೂಡಿಸು_2->src[1];
    ಆಗ (ಮುಖವಾಡ->op == GGML_OP_RESHAPE || ಮುಖವಾಡ->op == GGML_OP_CPY) {
        ಮುಖವಾಡ = ಮುಖವಾಡ->src[0];
    }

    ಸ್ಥಿರಾಂಕ uint32_t n_tps    = scores->ne[0];
    ಸ್ಥಿರಾಂಕ uint32_t n_ಖಂಡಗಳು = scores->ne[1];
    ಸ್ಥಿರಾಂಕ uint32_t n_ಹರಿವು = scores->ne[2];
    ಸ್ಥಿರಾಂಕ uint32_t n_kv     = cell_blk->ne[0];
    ಸ್ಥಿರಾಂಕ uint32_t ಅಗಲ    = ತುದಿ_k->ne[0];
    ಸ್ಥಿರಾಂಕ uint32_t nrows    = n_tps * n_ಹರಿವು;

    vk_pipeline pipeline = ctx->device->pipeline_topk_radix_qsa;
    GGML_ASSERT(pipeline != ಶೂನ್ಯಸೂಚಕ);

    // scratch holds the gathered+masked input, materialized once and reused across passes
    ಸ್ಥಿರಾಂಕ size_t scratch_ಗಾತ್ರ = size_t{ n_kv } * nrows * ಗಾತ್ರ(ದಶಮಾಂಶ);
    ಆದರೆ (ctx->prealloc_ಗಾತ್ರ_x < scratch_ಗಾತ್ರ) {
        ctx->prealloc_ಗಾತ್ರ_x = scratch_ಗಾತ್ರ;
        ggml_vk_preallocate_buffers(ctx, subctx);
    }
    ಆದರೆ (ctx->prealloc_x_need_sync) {
        ggml_vk_sync_buffers(ctx, subctx);
    }

    vk_op_topk_radix_ತಳ್ಳು_constants pc { n_kv, ಅಗಲ, nrows, n_tps, n_ಖಂಡಗಳು, n_ಹರಿವು };
    std::array<uint32_t, 3> elements {
        pipeline->wg_denoms[0],
        std::min(nrows, ctx->device->properties.limits.max_Compute_Work_ಗುಂಪು_ಎಣಿಕೆ[1]),
        1,
    };
    vk_subbuffer scratch_buf { ctx->prealloc_x, 0, ctx->prealloc_x->size };
    ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);
    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline,
        { ggml_vk_tensor_subbuffer(ctx, scores), ggml_vk_tensor_subbuffer(ctx, ತುದಿ_k),
          ggml_vk_tensor_subbuffer(ctx, cell_blk), ggml_vk_tensor_subbuffer(ctx, ಮುಖವಾಡ),
          scratch_buf }, pc, elements);
    ctx->prealloc_x_need_sync = ನಿಜ;
}

ಶೂನ್ಯ ggml_vk_ಮೊತ್ತ(ggml_backend_vk_context * ctx, vk_context& subctx, ಸ್ಥಿರಾಂಕ ggml_tensor * src0, ggml_tensor * dst) {
    vk_op_ಮೊತ್ತ_ಅಡ್ಡಸಾಲುಗಳು_ತಳ್ಳು_constants p = vk_op_ಮೊತ್ತ_ಅಡ್ಡಸಾಲುಗಳು_ತಳ್ಳು_constants_init(src0, dst, ggml_nelements(src0));
    ggml_vk_op_f32(ctx, subctx, src0, ಶೂನ್ಯಸೂಚಕ, ಶೂನ್ಯಸೂಚಕ, ಶೂನ್ಯಸೂಚಕ, dst, GGML_OP_ಮೊತ್ತ, p);
}

ಶೂನ್ಯ ggml_vk_ಮೊತ್ತ_ಅಡ್ಡಸಾಲುಗಳು(ggml_backend_vk_context * ctx, vk_context& subctx, ಸ್ಥಿರಾಂಕ ggml_tensor * src0, ggml_tensor * dst) {
    vk_op_ಮೊತ್ತ_ಅಡ್ಡಸಾಲುಗಳು_ತಳ್ಳು_constants p = vk_op_ಮೊತ್ತ_ಅಡ್ಡಸಾಲುಗಳು_ತಳ್ಳು_constants_init(src0, dst, src0->ne[0]);
    ggml_vk_op_f32(ctx, subctx, src0, ಶೂನ್ಯಸೂಚಕ, ಶೂನ್ಯಸೂಚಕ, ಶೂನ್ಯಸೂಚಕ, dst, GGML_OP_ಮೊತ್ತ_ಅಡ್ಡಸಾಲುಗಳು, p);
}

ಶೂನ್ಯ ggml_vk_ಸರಾಸರಿ(ggml_backend_vk_context * ctx, vk_context& subctx, ಸ್ಥಿರಾಂಕ ggml_tensor * src0, ggml_tensor * dst) {
    vk_op_ಮೊತ್ತ_ಅಡ್ಡಸಾಲುಗಳು_ತಳ್ಳು_constants p = vk_op_ಮೊತ್ತ_ಅಡ್ಡಸಾಲುಗಳು_ತಳ್ಳು_constants_init(src0, dst, src0->ne[0]);
    p.ತೂಕ = 1.0f / (ದಶಮಾಂಶ)src0->ne[0];
    ggml_vk_op_f32(ctx, subctx, src0, ಶೂನ್ಯಸೂಚಕ, ಶೂನ್ಯಸೂಚಕ, ಶೂನ್ಯಸೂಚಕ, dst, GGML_OP_ಸರಾಸರಿ, p);
}

ಶೂನ್ಯ ggml_vk_cumsum(ggml_backend_vk_context * ctx, vk_context& subctx, ಸ್ಥಿರಾಂಕ ggml_tensor * src0, ggml_tensor * dst) {
    vk_op_ಮೊತ್ತ_ಅಡ್ಡಸಾಲುಗಳು_ತಳ್ಳು_constants pc = vk_op_ಮೊತ್ತ_ಅಡ್ಡಸಾಲುಗಳು_ತಳ್ಳು_constants_init(src0, dst, src0->ne[0]);
    // Use the single pass shader when the rows are small or there are enough rows to fill the GPU.
    // For fewer, larger rows, use the multipass shader to spread each row across SMs.
    ಆದರೆ (dst->ne[0] <= 4096 || ggml_nrows(dst) >= ctx->device->shader_core_ಎಣಿಕೆ) {
        ggml_vk_op_f32(ctx, subctx, src0, ಶೂನ್ಯಸೂಚಕ, ಶೂನ್ಯಸೂಚಕ, ಶೂನ್ಯಸೂಚಕ, dst, GGML_OP_CUMSUM, pc);
        ಹಿಂತಿರುಗಿ;
    }

    // First pass computes partial sums within a block, and stores the last partial
    // to the temp buffer. Second pass sums the block partials from the temp buffer
    // and adds that to the result of the first pass.
    vk_pipeline pipeline1 = ctx->device->pipeline_cumsum_multipass1_f32;
    vk_pipeline pipeline2 = ctx->device->pipeline_cumsum_multipass2_f32;
    GGML_ASSERT(pipeline1 != ಶೂನ್ಯಸೂಚಕ && pipeline2 != ಶೂನ್ಯಸೂಚಕ);

    ggml_pipeline_request_descriptor_sets(ctx, pipeline1, 1);
    ggml_pipeline_request_descriptor_sets(ctx, pipeline2, 1);

    std::array<uint32_t, 3> elements;

    elements[0] = dst->ne[0];
    elements[1] = (uint32_t)ggml_nrows(dst);
    elements[2] = 1;

    size_t ತಾತ್ಕಾಲಿಕ_ಗಾತ್ರ = ಗಾತ್ರ(ದಶಮಾಂಶ) * elements[0] * ggml_nrows(dst);

    ಆದರೆ (ctx->prealloc_ಗಾತ್ರ_ವಿಭಜಿಸು_k < ತಾತ್ಕಾಲಿಕ_ಗಾತ್ರ) {
        ctx->prealloc_ಗಾತ್ರ_ವಿಭಜಿಸು_k = ತಾತ್ಕಾಲಿಕ_ಗಾತ್ರ;
        ggml_vk_preallocate_buffers(ctx, subctx);
    }

    vk_subbuffer src_buf = ggml_vk_tensor_subbuffer(ctx, src0);
    vk_subbuffer dst_buf = ggml_vk_tensor_subbuffer(ctx, dst);
    vk_subbuffer ತಾತ್ಕಾಲಿಕ_buf = ggml_vk_subbuffer(ctx, ctx->prealloc_ವಿಭಜಿಸು_k, 0);

    ಆದರೆ (ctx->prealloc_ವಿಭಜಿಸು_k_need_sync) {
        ggml_vk_sync_buffers(ctx, subctx);
    }

    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline1, {src_buf, dst_buf, ತಾತ್ಕಾಲಿಕ_buf}, pc, elements);
    ggml_vk_sync_buffers(ctx, subctx);
    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline2, {src_buf, dst_buf, ತಾತ್ಕಾಲಿಕ_buf}, pc, elements);

    ctx->prealloc_ವಿಭಜಿಸು_k_need_sync = ನಿಜ;
}

ಸ್ಥಿರ std::array<uint32_t, 3> ggml_vk_nrows_elements(uint32_t nr) {
    ಆದರೆ (nr > 262144) {
        ಹಿಂತಿರುಗಿ { 512, 512, CEIL_DIV(nr, 262144) };
    }
    ಆದರೆ (nr > 512) {
        ಹಿಂತಿರುಗಿ { 512, CEIL_DIV(nr, 512), 1 };
    }
    ಹಿಂತಿರುಗಿ { nr, 1, 1 };
}

ಶೂನ್ಯ ggml_vk_cross_entropy_loss(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst) {
    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = dst->src[0];
    ಸ್ಥಿರಾಂಕ ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(src1));
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_are_same_shape(src0, src1));
    GGML_ASSERT(ggml_is_scalar(dst));

    ಸ್ಥಿರಾಂಕ uint32_t nclasses = (uint32_t)src0->ne[0];
    ಸ್ಥಿರಾಂಕ uint32_t nrows    = (uint32_t)ggml_nrows(src0);

    vk_pipeline pipeline = ggml_vk_op_get_pipeline(ctx, src0, src1, ಶೂನ್ಯಸೂಚಕ, dst, GGML_OP_CROSS_ENTROPY_LOSS);
    GGML_ASSERT(pipeline != ಶೂನ್ಯಸೂಚಕ);

    ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);
    ggml_pipeline_request_descriptor_sets(ctx, ctx->device->pipeline_ಮೊತ್ತ_ಅಡ್ಡಸಾಲುಗಳು_f32, 1);

    vk_subbuffer src0_buf = ggml_vk_tensor_subbuffer(ctx, src0);
    vk_subbuffer src1_buf = ggml_vk_tensor_subbuffer(ctx, src1);
    vk_subbuffer dst_buf  = ggml_vk_tensor_subbuffer(ctx, dst, ನಿಜ);

    ಸ್ಥಿರಾಂಕ vk_op_ತಳ್ಳು_constants pc = { nclasses, nrows, 0.0f, 0.0f, 0.0f, 0.0f };

    ಸ್ಥಿರಾಂಕ size_t tmp_ಗಾತ್ರ = (size_t)nrows * ಗಾತ್ರ(ದಶಮಾಂಶ);
    ಆದರೆ (ctx->prealloc_ಗಾತ್ರ_x < tmp_ಗಾತ್ರ) {
        ctx->prealloc_ಗಾತ್ರ_x = tmp_ಗಾತ್ರ;
        ggml_vk_preallocate_buffers(ctx, subctx);
    }
    ಆದರೆ (ctx->prealloc_x_need_sync) {
        ggml_vk_sync_buffers(ctx, subctx);
    }

    vk_subbuffer tmp_buf = { ctx->prealloc_x, 0, tmp_ಗಾತ್ರ };
    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { src0_buf, src1_buf, tmp_buf }, pc, ggml_vk_nrows_elements(nrows));
    ggml_vk_sync_buffers(ctx, subctx);

    vk_op_ಮೊತ್ತ_ಅಡ್ಡಸಾಲುಗಳು_ತಳ್ಳು_constants sp = {};
    sp.n_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ = nrows;
    sp.ne01 = 1;
    sp.ne02 = 1;
    sp.ತೂಕ = 1.0f;
    init_pushconst_fastdiv(sp);
    sp.misalign_offsets = get_misalign_ಬೈಟುಗಳು(ctx, dst) / ggml_type_ಗಾತ್ರ(dst->type);

    ggml_vk_dispatch_pipeline(ctx, subctx, ctx->device->pipeline_ಮೊತ್ತ_ಅಡ್ಡಸಾಲುಗಳು_f32, { tmp_buf, dst_buf }, sp, { 1, 1, 1 });
    ctx->prealloc_x_need_sync = ನಿಜ;
}

ಶೂನ್ಯ ggml_vk_cross_entropy_loss_ಹಿಂಭಾಗ(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst) {
    ಸ್ಥಿರಾಂಕ ggml_tensor * grad   = dst->src[0];
    ಸ್ಥಿರಾಂಕ ggml_tensor * logits = dst->src[1];
    ಸ್ಥಿರಾಂಕ ggml_tensor * labels = dst->src[2];

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

    ಸ್ಥಿರಾಂಕ uint32_t nclasses = (uint32_t)logits->ne[0];
    ಸ್ಥಿರಾಂಕ uint32_t nrows    = (uint32_t)ggml_nrows(logits);

    vk_pipeline pipeline = ggml_vk_op_get_pipeline(ctx, grad, logits, labels, dst, GGML_OP_CROSS_ENTROPY_LOSS_ಹಿಂಭಾಗ);
    GGML_ASSERT(pipeline != ಶೂನ್ಯಸೂಚಕ);

    ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);

    vk_subbuffer grad_buf   = ggml_vk_tensor_subbuffer(ctx, grad);
    vk_subbuffer logits_buf = ggml_vk_tensor_subbuffer(ctx, logits);
    vk_subbuffer labels_buf = ggml_vk_tensor_subbuffer(ctx, labels);
    vk_subbuffer dst_buf    = ggml_vk_tensor_subbuffer(ctx, dst);

    ಸ್ಥಿರಾಂಕ vk_op_ತಳ್ಳು_constants pc = { nclasses, nrows, 0.0f, 0.0f, 0.0f, 0.0f };
    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { grad_buf, logits_buf, labels_buf, dst_buf }, pc, ggml_vk_nrows_elements(nrows));
}

ಶೂನ್ಯ ggml_vk_argmax(ggml_backend_vk_context * ctx, vk_context& subctx, ಸ್ಥಿರಾಂಕ ggml_tensor * src0, ggml_tensor * dst) {
    ggml_vk_op_f32<vk_op_ತಳ್ಳು_constants>(ctx, subctx, src0, ಶೂನ್ಯಸೂಚಕ, ಶೂನ್ಯಸೂಚಕ, ಶೂನ್ಯಸೂಚಕ, dst, GGML_OP_ARGMAX, { (uint32_t)src0->ne[0], (uint32_t)src0->ne[1], 0.0f, 0.0f, 0.0f, 0.0f });
}

ಶೂನ್ಯ ggml_vk_ಎಣಿಕೆ_equal(ggml_backend_vk_context * ctx, vk_context& subctx, ಸ್ಥಿರಾಂಕ ggml_tensor * src0, ಸ್ಥಿರಾಂಕ ggml_tensor * src1, ggml_tensor * dst) {
    ggml_vk_op_f32<vk_op_ತಳ್ಳು_constants>(ctx, subctx, src0, src1, ಶೂನ್ಯಸೂಚಕ, ಶೂನ್ಯಸೂಚಕ, dst, GGML_OP_ಎಣಿಕೆ_EQUAL, { (uint32_t)ggml_nelements(src0), 0, 0.0f, 0.0f, 0.0f, 0.0f });
}

ಶೂನ್ಯ ggml_vk_solve_tri(ggml_backend_vk_context * ctx, vk_context& subctx, ಸ್ಥಿರಾಂಕ ggml_tensor * src0, ಸ್ಥಿರಾಂಕ ggml_tensor * src1, ggml_tensor * dst) {
    ಸ್ಥಿರಾಂಕ uint32_t src0_type_ಗಾತ್ರ = ggml_type_ಗಾತ್ರ(src0->type);
    ಸ್ಥಿರಾಂಕ uint32_t src1_type_ಗಾತ್ರ = ggml_type_ಗಾತ್ರ(src1->type);
    ಸ್ಥಿರಾಂಕ uint32_t dst_type_ಗಾತ್ರ = ggml_type_ಗಾತ್ರ(dst->type);

    ggml_vk_op_f32<vk_op_ದ್ವಿಮಾನ_ತಳ್ಳು_constants>(ctx, subctx, src0, src1, ಶೂನ್ಯಸೂಚಕ, ಶೂನ್ಯಸೂಚಕ, dst, GGML_OP_SOLVE_TRI, {
        (uint32_t)ggml_nelements(src0),
        (uint32_t)src0->ne[0], (uint32_t)src0->ne[1], (uint32_t)src0->ne[2],(uint32_t)src0->ne[3], (uint32_t)src0->nb[0] / src0_type_ಗಾತ್ರ, (uint32_t)src0->nb[1] / src0_type_ಗಾತ್ರ, (uint32_t)src0->nb[2] / src0_type_ಗಾತ್ರ, (uint32_t)src0->nb[3] / src0_type_ಗಾತ್ರ,
        (uint32_t)src1->ne[0], (uint32_t)src1->ne[1], (uint32_t)src1->ne[2],(uint32_t)src1->ne[3], (uint32_t)src1->nb[0] / src1_type_ಗಾತ್ರ, (uint32_t)src1->nb[1] / src1_type_ಗಾತ್ರ, (uint32_t)src1->nb[2] / src1_type_ಗಾತ್ರ, (uint32_t)src1->nb[3] / src1_type_ಗಾತ್ರ,
        (uint32_t) dst->ne[0], (uint32_t) dst->ne[1], (uint32_t) dst->ne[2],(uint32_t) dst->ne[3], (uint32_t) dst->nb[0] /  dst_type_ಗಾತ್ರ, (uint32_t) dst->nb[1] /  dst_type_ಗಾತ್ರ, (uint32_t) dst->nb[2] /  dst_type_ಗಾತ್ರ, (uint32_t) dst->nb[3] /  dst_type_ಗಾತ್ರ,
        0,
        0.0f, 0.0f, 0,
    });
}

ಶೂನ್ಯ ggml_vk_im2col(ggml_backend_vk_context * ctx, vk_context& subctx, ಸ್ಥಿರಾಂಕ ggml_tensor * src0, ಸ್ಥಿರಾಂಕ ggml_tensor * src1, ggml_tensor * dst) {
    ಸ್ಥಿರಾಂಕ int32_t s0 = dst->op_params[0];
    ಸ್ಥಿರಾಂಕ int32_t s1 = dst->op_params[1];
    ಸ್ಥಿರಾಂಕ int32_t p0 = dst->op_params[2];
    ಸ್ಥಿರಾಂಕ int32_t p1 = dst->op_params[3];
    ಸ್ಥಿರಾಂಕ int32_t d0 = dst->op_params[4];
    ಸ್ಥಿರಾಂಕ int32_t d1 = dst->op_params[5];

    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ is_2D = dst->op_params[6] == 1;

    ಸ್ಥಿರಾಂಕ uint32_t IC = src1->ne[is_2D ? 2 : 1];
    ಸ್ಥಿರಾಂಕ uint32_t IH = is_2D ? src1->ne[1] : 1;
    ಸ್ಥಿರಾಂಕ uint32_t IW =         src1->ne[0];

    ಸ್ಥಿರಾಂಕ uint32_t KH = is_2D ? src0->ne[1] : 1;
    ಸ್ಥಿರಾಂಕ uint32_t KW =         src0->ne[0];

    ಸ್ಥಿರಾಂಕ uint32_t OH = is_2D ? dst->ne[2] : 1;
    ಸ್ಥಿರಾಂಕ uint32_t OW =         dst->ne[1];

    ಸ್ಥಿರಾಂಕ uint32_t ಸ್ಥಾನಾಂತರ_delta = src1->nb[is_2D ? 2 : 1] / 4; // nb is byte offset, src is type float32
    ಸ್ಥಿರಾಂಕ uint32_t batch_ಸ್ಥಾನಾಂತರ = src1->nb[is_2D ? 3 : 2] / 4; // nb is byte offset, src is type float32

    ಸ್ಥಿರಾಂಕ uint32_t batch = src1->ne[is_2D ? 3 : 2];

    ಸ್ಥಿರಾಂಕ ggml_backend_vk_ಬಫರ್_context * d_buf_ctx = (ggml_backend_vk_ಬಫರ್_context *)dst->ಬಫರ್->context;
    ಸ್ಥಿರಾಂಕ vk_ಬಫರ್ d_buf = d_buf_ctx->dev_ಬಫರ್;

    ಸ್ಥಿರಾಂಕ vk::Device_ವಿಳಾಸ dst_addr = d_buf->bda_addr + vk_tensor_ಸ್ಥಾನಾಂತರ(dst) + dst->view_offs;

    ggml_vk_op_f32<vk_op_im2col_ತಳ್ಳು_constants>(ctx, subctx, src0, src1, ಶೂನ್ಯಸೂಚಕ, ಶೂನ್ಯಸೂಚಕ, dst, GGML_OP_IM2_ಲಂಬಸಾಲು, {
        dst_addr,
        batch_ಸ್ಥಾನಾಂತರ, ಸ್ಥಾನಾಂತರ_delta,
        IC, IW, IH, OW, OH, KW, KH,
        OH * batch,
        IC * KH * KW,
        s0, s1, p0, p1, d0, d1, batch * IC
    });
}

ಶೂನ್ಯ ggml_vk_im2col_3d(ggml_backend_vk_context * ctx, vk_context& subctx, ಸ್ಥಿರಾಂಕ ggml_tensor * src0, ಸ್ಥಿರಾಂಕ ggml_tensor * src1, ggml_tensor * dst) {
    GGML_TENSOR_ದ್ವಿಮಾನ_OP_LOCALS

    ಸ್ಥಿರಾಂಕ int32_t s0 = ((ಸ್ಥಿರಾಂಕ int32_t *)(dst->op_params))[0];
    ಸ್ಥಿರಾಂಕ int32_t s1 = ((ಸ್ಥಿರಾಂಕ int32_t *)(dst->op_params))[1];
    ಸ್ಥಿರಾಂಕ int32_t s2 = ((ಸ್ಥಿರಾಂಕ int32_t *)(dst->op_params))[2];
    ಸ್ಥಿರಾಂಕ int32_t p0 = ((ಸ್ಥಿರಾಂಕ int32_t *)(dst->op_params))[3];
    ಸ್ಥಿರಾಂಕ int32_t p1 = ((ಸ್ಥಿರಾಂಕ int32_t *)(dst->op_params))[4];
    ಸ್ಥಿರಾಂಕ int32_t p2 = ((ಸ್ಥಿರಾಂಕ int32_t *)(dst->op_params))[5];
    ಸ್ಥಿರಾಂಕ int32_t d0 = ((ಸ್ಥಿರಾಂಕ int32_t *)(dst->op_params))[6];
    ಸ್ಥಿರಾಂಕ int32_t d1 = ((ಸ್ಥಿರಾಂಕ int32_t *)(dst->op_params))[7];
    ಸ್ಥಿರಾಂಕ int32_t d2 = ((ಸ್ಥಿರಾಂಕ int32_t *)(dst->op_params))[8];
    ಸ್ಥಿರಾಂಕ int32_t IC = ((ಸ್ಥಿರಾಂಕ int32_t *)(dst->op_params))[9];

    ಸ್ಥಿರಾಂಕ int64_t N  = ne13 / IC;
    ಸ್ಥಿರಾಂಕ int64_t ID = ne12;
    ಸ್ಥಿರಾಂಕ int64_t IH = ne11;
    ಸ್ಥಿರಾಂಕ int64_t IW = ne10;

    ಸ್ಥಿರಾಂಕ int64_t KD = ne02;
    ಸ್ಥಿರಾಂಕ int64_t KH = ne01;
    ಸ್ಥಿರಾಂಕ int64_t KW = ne00;

    ಸ್ಥಿರಾಂಕ int64_t OD = ne3 / N;
    ಸ್ಥಿರಾಂಕ int64_t OH = ne2;
    ಸ್ಥಿರಾಂಕ int64_t OW = ne1;

    ಸ್ಥಿರಾಂಕ ggml_backend_vk_ಬಫರ್_context * d_buf_ctx = (ggml_backend_vk_ಬಫರ್_context *)dst->ಬಫರ್->context;
    ಸ್ಥಿರಾಂಕ vk_ಬಫರ್ d_buf = d_buf_ctx->dev_ಬಫರ್;

    ಸ್ಥಿರಾಂಕ vk::Device_ವಿಳಾಸ dst_addr = d_buf->bda_addr + vk_tensor_ಸ್ಥಾನಾಂತರ(dst) + dst->view_offs;

    vk_op_im2col_3d_ತಳ್ಳು_constants pc {};

    pc.dst_addr = dst_addr;
    pc.nb10 = nb10 / ggml_type_ಗಾತ್ರ(src1->type);
    pc.nb11 = nb11 / ggml_type_ಗಾತ್ರ(src1->type);
    pc.nb12 = nb12 / ggml_type_ಗಾತ್ರ(src1->type);
    pc.nb13 = nb13 / ggml_type_ಗಾತ್ರ(src1->type);
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

    ggml_vk_op_f32<vk_op_im2col_3d_ತಳ್ಳು_constants>(ctx, subctx, src0, src1, ಶೂನ್ಯಸೂಚಕ, ಶೂನ್ಯಸೂಚಕ, dst, GGML_OP_IM2_ಲಂಬಸಾಲು_3_D, std::move(pc));
}

ಶೂನ್ಯ ggml_vk_timestep_embedding(ggml_backend_vk_context * ctx, vk_context& subctx, ಸ್ಥಿರಾಂಕ ggml_tensor * src0, ggml_tensor * dst) {
    ಸ್ಥಿರಾಂಕ uint32_t dim = dst->op_params[0];
    ಸ್ಥಿರಾಂಕ uint32_t max_period = dst->op_params[1];
    ಸ್ಥಿರಾಂಕ uint32_t nb1 = dst->nb[1] / ggml_type_ಗಾತ್ರ(dst->type);

    ggml_vk_op_f32<vk_op_timestep_embedding_ತಳ್ಳು_constants>(ctx, subctx, src0, ಶೂನ್ಯಸೂಚಕ, ಶೂನ್ಯಸೂಚಕ, ಶೂನ್ಯಸೂಚಕ, dst, GGML_OP_TIMESTEP_EMBEDDING, {
        nb1, dim, max_period,
    });
}

ಶೂನ್ಯ ggml_vk_conv_transpose_1d(ggml_backend_vk_context * ctx, vk_context& subctx, ಸ್ಥಿರಾಂಕ ggml_tensor * src0, ಸ್ಥಿರಾಂಕ ggml_tensor * src1, ggml_tensor * dst) {
    // src0: (K, Cout, Cin, 1) -- kernel
    // src1: (L, Cin, 1, 1) -- input
    // dst: (*, Cout, 1, 1)

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_TENSOR_ದ್ವಿಮಾನ_OP_LOCALS

    GGML_ASSERT(nb00 == ಗಾತ್ರ(ದಶಮಾಂಶ));
    GGML_ASSERT(nb10 == ಗಾತ್ರ(ದಶಮಾಂಶ));

    ಸ್ಥಿರಾಂಕ int32_t s0 = dst->op_params[0];

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

    ggml_vk_op_f32(ctx, subctx, src0, src1, ಶೂನ್ಯಸೂಚಕ, ಶೂನ್ಯಸೂಚಕ, dst, GGML_OP_CONV_TRANSPOSE_1D, std::move(p));
}

ಶೂನ್ಯ ggml_vk_col2im_1d(ggml_backend_vk_context * ctx, vk_context& subctx, ಸ್ಥಿರಾಂಕ ggml_tensor * src0, ggml_tensor * dst) {
    // src0: [K_OC, T_in] columns from matmul
    // dst:  [T_out, OC]

    ಸ್ಥಿರಾಂಕ int32_t stride = dst->op_params[0];
    ಸ್ಥಿರಾಂಕ int32_t oc     = dst->op_params[1];
    ಸ್ಥಿರಾಂಕ int32_t p0     = dst->op_params[2];

    ಸ್ಥಿರಾಂಕ uint32_t K_OC  = static_cast<uint32_t>(src0->ne[0]);
    ಸ್ಥಿರಾಂಕ uint32_t T_ಒಳ  = static_cast<uint32_t>(src0->ne[1]);
    ಸ್ಥಿರಾಂಕ uint32_t T_ಹೊರಹೋಗು = static_cast<uint32_t>(dst->ne[0]);
    ಸ್ಥಿರಾಂಕ uint32_t OC    = static_cast<uint32_t>(oc);
    ಸ್ಥಿರಾಂಕ uint32_t K     = K_OC / OC;

    vk_op_col2im_1d_ತಳ್ಳು_constants p{};
    p.T_ಹೊರಹೋಗು  = T_ಹೊರಹೋಗು;
    p.OC     = OC;
    p.K_OC   = K_OC;
    p.T_ಒಳ   = T_ಒಳ;
    p.K      = K;
    p.stride = stride;
    p.p0     = p0;

    ggml_vk_op_f32(ctx, subctx, src0, ಶೂನ್ಯಸೂಚಕ, ಶೂನ್ಯಸೂಚಕ, ಶೂನ್ಯಸೂಚಕ, dst, GGML_OP_COL2IM_1D, std::move(p));
}

ಶೂನ್ಯ ggml_vk_snake_dispatch_fused(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_cgraph * cgraph, ಪೂರ್ಣಾಂಕ ಗಂಟು_idx) {
    ಸ್ಥಿರಾಂಕ ggml_tensor * mul0 = cgraph->nodes[ಗಂಟು_idx + 0];
    ಸ್ಥಿರಾಂಕ ggml_tensor * sqr  = cgraph->nodes[ಗಂಟು_idx + 2];
    ಸ್ಥಿರಾಂಕ ggml_tensor * mul1 = cgraph->nodes[ಗಂಟು_idx + 3];
    ggml_tensor *       ಕೂಡಿಸು_2  = cgraph->nodes[ಗಂಟು_idx + 4];

    // x carries the full activation shape, a is the broadcast operand
    ಸ್ಥಿರಾಂಕ ggml_tensor * x = ggml_are_same_shape(mul0, mul0->src[0]) ? mul0->src[0] : mul0->src[1];
    ಸ್ಥಿರಾಂಕ ggml_tensor * a = (x == mul0->src[0]) ? mul0->src[1] : mul0->src[0];

    // mul1 reads sqr and inv_b in either operand order
    ಸ್ಥಿರಾಂಕ ggml_tensor * inv_b = (mul1->src[0] == sqr) ? mul1->src[1] : mul1->src[0];

    vk_pipeline pipeline = ಶೂನ್ಯಸೂಚಕ;
    ಆಯ್ಕೆ (x->type) {
        ಸ್ಥಿತಿ GGML_TYPE_F32:  pipeline = ctx->device->pipeline_snake_f32;  ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_F16:  pipeline = ctx->device->pipeline_snake_f16;  ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_BF16: pipeline = ctx->device->pipeline_snake_bf16; ನಿಲ್ಲಿಸು;
        ಪೂರ್ವನಿಯೋಜಿತ:             GGML_ABORT("unsupported type");
    }
    ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);

    vk_subbuffer x_buf     = ggml_vk_tensor_subbuffer(ctx, x);
    vk_subbuffer a_buf     = ggml_vk_tensor_subbuffer(ctx, a);
    vk_subbuffer inv_b_buf = ggml_vk_tensor_subbuffer(ctx, inv_b);
    vk_subbuffer dst_buf   = ggml_vk_tensor_subbuffer(ctx, ಕೂಡಿಸು_2);

    vk_op_snake_ತಳ್ಳು_constants pc{};
    pc.ne0 = static_cast<uint32_t>(x->ne[0]);
    pc.ne1 = static_cast<uint32_t>(x->ne[1]);

    std::array<uint32_t, 3> elements = { pc.ne0, pc.ne1, 1 };
    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { x_buf, a_buf, inv_b_buf, dst_buf }, pc, elements);
}

ಶೂನ್ಯ ggml_vk_pool_1d(ggml_backend_vk_context * ctx, vk_context& subctx, ಸ್ಥಿರಾಂಕ ggml_tensor * src0, ggml_tensor * dst) {
    uint32_t op = static_cast<uint32_t>(dst->op_params[0]);
    ಸ್ಥಿರಾಂಕ int32_t k0 = dst->op_params[1];
    ಸ್ಥಿರಾಂಕ int32_t s0 = dst->op_params[2];
    ಸ್ಥಿರಾಂಕ int32_t p0 = dst->op_params[3];

    ಸ್ಥಿರಾಂಕ uint32_t IL = src0->ne[0];

    ಸ್ಥಿರಾಂಕ uint32_t N = dst->ne[3] * dst->ne[2];

    ಸ್ಥಿರಾಂಕ uint32_t OC = dst->ne[1];
    ಸ್ಥಿರಾಂಕ uint32_t OL = dst->ne[0];

    ಸ್ಥಿರಾಂಕ uint32_t parallel_elements = N * OC * OL;

    ggml_vk_op_f32<vk_op_pool1d_ತಳ್ಳು_constants>(ctx, subctx, src0, ಶೂನ್ಯಸೂಚಕ, ಶೂನ್ಯಸೂಚಕ, ಶೂನ್ಯಸೂಚಕ, dst, GGML_OP_POOL_1D, {
        IL, OL, OC,
        parallel_elements,
        op,
        k0, s0, p0,
    });
}

ಶೂನ್ಯ ggml_vk_pool_2d(ggml_backend_vk_context * ctx, vk_context& subctx, ಸ್ಥಿರಾಂಕ ggml_tensor * src0, ggml_tensor * dst) {
    uint32_t op = static_cast<uint32_t>(dst->op_params[0]);
    ಸ್ಥಿರಾಂಕ int32_t k1 = dst->op_params[1];
    ಸ್ಥಿರಾಂಕ int32_t k0 = dst->op_params[2];
    ಸ್ಥಿರಾಂಕ int32_t s1 = dst->op_params[3];
    ಸ್ಥಿರಾಂಕ int32_t s0 = dst->op_params[4];
    ಸ್ಥಿರಾಂಕ int32_t p1 = dst->op_params[5];
    ಸ್ಥಿರಾಂಕ int32_t p0 = dst->op_params[6];

    ಸ್ಥಿರಾಂಕ uint32_t IH = src0->ne[1];
    ಸ್ಥಿರಾಂಕ uint32_t IW = src0->ne[0];

    ಸ್ಥಿರಾಂಕ uint32_t N = dst->ne[3];

    ಸ್ಥಿರಾಂಕ uint32_t OC = dst->ne[2];
    ಸ್ಥಿರಾಂಕ uint32_t OH = dst->ne[1];
    ಸ್ಥಿರಾಂಕ uint32_t OW = dst->ne[0];

    ಸ್ಥಿರಾಂಕ uint32_t parallel_elements = N * OC * OH * OW;

    ggml_vk_op_f32<vk_op_pool2d_ತಳ್ಳು_constants>(ctx, subctx, src0, ಶೂನ್ಯಸೂಚಕ, ಶೂನ್ಯಸೂಚಕ, ಶೂನ್ಯಸೂಚಕ, dst, GGML_OP_POOL_2D, {
        IW, IH, OW, OH, OC,
        parallel_elements,
        op,
        k0, k1, s0, s1, p0, p1,
    });
}

ಶೂನ್ಯ ggml_vk_conv_2d(ggml_backend_vk_context * ctx, vk_context & subctx, ಸ್ಥಿರಾಂಕ ggml_tensor * src0,
                            ಸ್ಥಿರಾಂಕ ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    GGML_TENSOR_ದ್ವಿಮಾನ_OP_LOCALS
    GGML_ASSERT(nb00 == ಗಾತ್ರ(ದಶಮಾಂಶ) || nb00 == ಗಾತ್ರ(ggml_fp16_t));
    GGML_ASSERT(nb10 == ಗಾತ್ರ(ದಶಮಾಂಶ));
    GGML_ASSERT(nb0 == ಗಾತ್ರ(ದಶಮಾಂಶ));

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

    ggml_vk_op_f32(ctx, subctx, src0, src1, ಶೂನ್ಯಸೂಚಕ, ಶೂನ್ಯಸೂಚಕ, dst, dst->op, std::move(p));
}

ಶೂನ್ಯ ggml_vk_conv_3d(ggml_backend_vk_context * ctx, vk_context & subctx, ಸ್ಥಿರಾಂಕ ggml_tensor * src0,
                            ಸ್ಥಿರಾಂಕ ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    GGML_TENSOR_ದ್ವಿಮಾನ_OP_LOCALS
    GGML_ASSERT(nb00 == ಗಾತ್ರ(ದಶಮಾಂಶ) || nb00 == ಗಾತ್ರ(ggml_fp16_t));
    GGML_ASSERT(nb10 == ಗಾತ್ರ(ದಶಮಾಂಶ));
    GGML_ASSERT(nb0 == ಗಾತ್ರ(ದಶಮಾಂಶ));

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

    ggml_vk_op_f32(ctx, subctx, src0, src1, ಶೂನ್ಯಸೂಚಕ, ಶೂನ್ಯಸೂಚಕ, dst, GGML_OP_CONV_3D, std::move(p));
}

ಶೂನ್ಯ ggml_vk_conv_2d_dw(ggml_backend_vk_context * ctx, vk_context& subctx, ಸ್ಥಿರಾಂಕ ggml_tensor * src0, ಸ್ಥಿರಾಂಕ ggml_tensor * src1, ggml_tensor * dst) {
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

    ggml_vk_op_f32(ctx, subctx, src0, src1, ಶೂನ್ಯಸೂಚಕ, ಶೂನ್ಯಸೂಚಕ, dst, GGML_OP_CONV_2D_DW, std::move(p));
}

ಶೂನ್ಯ ggml_vk_leaky_relu(ggml_backend_vk_context * ctx, vk_context& subctx, ಸ್ಥಿರಾಂಕ ggml_tensor * src0, ggml_tensor * dst) {
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * op_params = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *)dst->op_params;
    vk_op_unary_ತಳ್ಳು_constants p = vk_op_unary_ತಳ್ಳು_constants_init(src0, dst);
    p.param1 = op_params[0];

    ggml_vk_op_f32(ctx, subctx, src0, ಶೂನ್ಯಸೂಚಕ, ಶೂನ್ಯಸೂಚಕ, ಶೂನ್ಯಸೂಚಕ, dst, GGML_OP_LEAKY_RELU, std::move(p));
}

ಶೂನ್ಯ ggml_vk_preallocate_buffers(ggml_backend_vk_context * ctx, vk_context subctx) {
#ಆದರೆ defined(GGML_VULKAN_ಸತತ_TESTS)
    ಸ್ಥಿರಾಂಕ std::vector<size_t> vals {
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
    ಸ್ಥಿರಾಂಕ size_t ಅಂಕೆ_it = 100;

    ggml_vk_test_dequant_matmul(ctx, 4096, 512, 4096, 2, ಅಂಕೆ_it, 1, 0, GGML_TYPE_Q4_0);
    ggml_vk_test_dequant_matmul(ctx, 4096, 512, 4096, 2, ಅಂಕೆ_it, 1, 1, GGML_TYPE_Q4_0);
    ggml_vk_test_dequant_matmul(ctx, 4096, 512, 4096, 2, ಅಂಕೆ_it, 1, 2, GGML_TYPE_Q4_0);

    ggml_vk_test_dequant_matmul(ctx, 4096, 512, 4096, 2, ಅಂಕೆ_it, 1, 0, GGML_TYPE_Q4_0, ನಿಜ);
    ggml_vk_test_dequant_matmul(ctx, 4096, 512, 4096, 2, ಅಂಕೆ_it, 1, 1, GGML_TYPE_Q4_0, ನಿಜ);
    ggml_vk_test_dequant_matmul(ctx, 4096, 512, 4096, 2, ಅಂಕೆ_it, 1, 2, GGML_TYPE_Q4_0, ನಿಜ);

    ggml_vk_test_dequant_matmul(ctx, 4096, 512, 4096, 2, ಅಂಕೆ_it, 1, 0, GGML_TYPE_Q8_0);
    ggml_vk_test_dequant_matmul(ctx, 4096, 512, 4096, 2, ಅಂಕೆ_it, 1, 1, GGML_TYPE_Q8_0);
    ggml_vk_test_dequant_matmul(ctx, 4096, 512, 4096, 2, ಅಂಕೆ_it, 1, 2, GGML_TYPE_Q8_0);

    ggml_vk_test_dequant_matmul(ctx, 4096, 512, 4096, 2, ಅಂಕೆ_it, 1, 0, GGML_TYPE_Q8_0, ನಿಜ);
    ggml_vk_test_dequant_matmul(ctx, 4096, 512, 4096, 2, ಅಂಕೆ_it, 1, 1, GGML_TYPE_Q8_0, ನಿಜ);
    ggml_vk_test_dequant_matmul(ctx, 4096, 512, 4096, 2, ಅಂಕೆ_it, 1, 2, GGML_TYPE_Q8_0, ನಿಜ);

    abort();

    ಗಾಗಿ (size_t i = 0; i < vals.size(); i += 3) {
        ggml_vk_test_matmul<ggml_fp16_t, ದಶಮಾಂಶ>(ctx, vals[i], vals[i + 1], vals[i + 2], 2, ಅಂಕೆ_it, 1, 0);
        ggml_vk_test_matmul<ggml_fp16_t, ದಶಮಾಂಶ>(ctx, vals[i], vals[i + 1], vals[i + 2], 2, ಅಂಕೆ_it, 1, 1);
        ggml_vk_test_matmul<ggml_fp16_t, ದಶಮಾಂಶ>(ctx, vals[i], vals[i + 1], vals[i + 2], 2, ಅಂಕೆ_it, 1, 2);
        std::cerr << '\n';
        ggml_vk_test_matmul<ggml_fp16_t, ದಶಮಾಂಶ>(ctx, vals[i], vals[i + 1], vals[i + 2], 2, ಅಂಕೆ_it, 2, 0);
        ggml_vk_test_matmul<ggml_fp16_t, ದಶಮಾಂಶ>(ctx, vals[i], vals[i + 1], vals[i + 2], 2, ಅಂಕೆ_it, 2, 1);
        ggml_vk_test_matmul<ggml_fp16_t, ದಶಮಾಂಶ>(ctx, vals[i], vals[i + 1], vals[i + 2], 2, ಅಂಕೆ_it, 2, 2);
        std::cerr << '\n';
        ggml_vk_test_matmul<ggml_fp16_t, ದಶಮಾಂಶ>(ctx, vals[i], vals[i + 1], vals[i + 2], 2, ಅಂಕೆ_it, 4, 0);
        ggml_vk_test_matmul<ggml_fp16_t, ದಶಮಾಂಶ>(ctx, vals[i], vals[i + 1], vals[i + 2], 2, ಅಂಕೆ_it, 4, 1);
        ggml_vk_test_matmul<ggml_fp16_t, ದಶಮಾಂಶ>(ctx, vals[i], vals[i + 1], vals[i + 2], 2, ಅಂಕೆ_it, 4, 2);
        std::cerr << '\n' << std::ಸಾಲು_ಅಂತ್ಯ;

        ಆದರೆ (vals[i + 2] % 32 == 0) {
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
            std::cerr << '\n' << std::ಸಾಲು_ಅಂತ್ಯ;
        }

        ಆದರೆ (vals[i + 2] % 256 == 0) {
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
            std::cerr << '\n' << std::ಸಾಲು_ಅಂತ್ಯ;
        }
    }

    GGML_ABORT("fatal error");
#ಆದರೆ_ಅಂತ್ಯ

    ಆದರೆ (subctx) {
        // Submit and wait for any pending work before reallocating the buffers
        ggml_vk_ctx_ಕೊನೆ(subctx);
        ggml_vk_submit(subctx, {});
        ctx->submit_pending = ನಿಜ;
        ggml_vk_synchronize(ctx);
        GGML_ASSERT(ctx->compute_ctx.expired());
        ggml_vk_ctx_begin(ctx->device, subctx);
        ctx->compute_ctx = subctx;
    }

    ಆದರೆ (ctx->prealloc_x == ಶೂನ್ಯಸೂಚಕ || (ctx->prealloc_ಗಾತ್ರ_x > 0 && ctx->prealloc_x->size < ctx->prealloc_ಗಾತ್ರ_x)) {
        VK_LOG_ಸ್ಮರಣೆ("ggml_vk_preallocate_buffers(x_size: " << ctx->prealloc_ಗಾತ್ರ_x << ")");
        // Resize buffer
        ಆದರೆ (ctx->prealloc_x != ಶೂನ್ಯಸೂಚಕ) {
            ggml_vk_destroy_ಬಫರ್(ctx->prealloc_x);
        }
        ctx->prealloc_x = ggml_vk_create_ಬಫರ್_device(ctx->device, ctx->prealloc_ಗಾತ್ರ_x);
    }
    ಆದರೆ (ctx->prealloc_y == ಶೂನ್ಯಸೂಚಕ || (ctx->prealloc_ಗಾತ್ರ_y > 0 && ctx->prealloc_y->size < ctx->prealloc_ಗಾತ್ರ_y)) {
        VK_LOG_ಸ್ಮರಣೆ("ggml_vk_preallocate_buffers(y_size: " << ctx->prealloc_ಗಾತ್ರ_y << ")");
        // Resize buffer
        ಆದರೆ (ctx->prealloc_y != ಶೂನ್ಯಸೂಚಕ) {
            ggml_vk_destroy_ಬಫರ್(ctx->prealloc_y);
        }
        ctx->prealloc_y = ggml_vk_create_ಬಫರ್_device(ctx->device, ctx->prealloc_ಗಾತ್ರ_y);
        ctx->prealloc_y_ಕೊನೆಯ_pipeline_ಬಳಸಿದ = ಶೂನ್ಯಸೂಚಕ;
        ctx->prealloc_y_ಕೊನೆಯ_tensor_ಬಳಸಿದ = ಶೂನ್ಯಸೂಚಕ;
        ctx->prealloc_y_ಕೊನೆಯ_k_padded = ಸುಳ್ಳು;
    }
    ಆದರೆ (ctx->prealloc_ವಿಭಜಿಸು_k == ಶೂನ್ಯಸೂಚಕ || (ctx->prealloc_ಗಾತ್ರ_ವಿಭಜಿಸು_k > 0 && ctx->prealloc_ವಿಭಜಿಸು_k->size < ctx->prealloc_ಗಾತ್ರ_ವಿಭಜಿಸು_k)) {
        VK_LOG_ಸ್ಮರಣೆ("ggml_vk_preallocate_buffers(split_k_size: " << ctx->prealloc_ಗಾತ್ರ_ವಿಭಜಿಸು_k << ")");
        // Resize buffer
        ಆದರೆ (ctx->prealloc_ವಿಭಜಿಸು_k != ಶೂನ್ಯಸೂಚಕ) {
            ggml_vk_destroy_ಬಫರ್(ctx->prealloc_ವಿಭಜಿಸು_k);
        }
        ctx->prealloc_ವಿಭಜಿಸು_k = ggml_vk_create_ಬಫರ್_device(ctx->device, ctx->prealloc_ಗಾತ್ರ_ವಿಭಜಿಸು_k);
    }
    ಆದರೆ (ctx->prealloc_ಕೂಡಿಸು_rms_partials == ಶೂನ್ಯಸೂಚಕ || (ctx->prealloc_ಗಾತ್ರ_ಕೂಡಿಸು_rms_partials > 0 && ctx->prealloc_ಕೂಡಿಸು_rms_partials->size < ctx->prealloc_ಗಾತ್ರ_ಕೂಡಿಸು_rms_partials)) {
        VK_LOG_ಸ್ಮರಣೆ("ggml_vk_preallocate_buffers(add_partials_size: " << ctx->prealloc_ಕೂಡಿಸು_rms_partials << ")");
        // Resize buffer
        ಆದರೆ (ctx->prealloc_ಕೂಡಿಸು_rms_partials != ಶೂನ್ಯಸೂಚಕ) {
            ggml_vk_destroy_ಬಫರ್(ctx->prealloc_ಕೂಡಿಸು_rms_partials);
        }
        ctx->prealloc_ಕೂಡಿಸು_rms_partials = ggml_vk_create_ಬಫರ್_device(ctx->device, ctx->prealloc_ಗಾತ್ರ_ಕೂಡಿಸು_rms_partials);
    }
}

ತಾರ್ಕಿಕ ggml_vk_build_ನಕ್ಷೆ(ggml_backend_vk_context * ctx, ggml_cgraph * cgraph, ಪೂರ್ಣಾಂಕ ಗಂಟು_idx, ggml_tensor *ಗಂಟು_begin, ಪೂರ್ಣಾಂಕ ಗಂಟು_idx_begin, ತಾರ್ಕಿಕ ಕೊನೆಯ_ಗಂಟು, ತಾರ್ಕಿಕ almost_ready, ತಾರ್ಕಿಕ submit){
    ggml_tensor * ಗಂಟು = cgraph->nodes[ಗಂಟು_idx];
    ಆದರೆ (ggml_is_ಖಾಲಿಗಳು(ಗಂಟು) || ggml_op_is_ಖಾಲಿಗಳು(ಗಂಟು->op) || !ಗಂಟು->ಬಫರ್) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
    ಆದರೆ ((ಗಂಟು->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    VK_LOG_DEBUG("ggml_vk_build_graph(" << ಗಂಟು << ", " << ggml_op_name(ಗಂಟು->op) << ")");
    ctx->semaphore_idx = 0;

    ggml_tensor * src0 = ಗಂಟು->src[0];
    ggml_tensor * src1 = ಗಂಟು->src[1];
    ggml_tensor * src2 = ಗಂಟು->src[2];
    ggml_tensor * src3 = ಗಂಟು->src[3];

    ಆದರೆ (ಗಂಟು->op == GGML_OP_ಕೂಡಿಸು) {
        ಪೂರ್ಣಾಂಕ ಮುಂದಿನ_ಗಂಟು_idx = ಗಂಟು_idx + 1 + ctx->ಅಂಕೆ_additional_fused_ops;
        ಆದರೆ (ಮುಂದಿನ_ಗಂಟು_idx < cgraph->n_nodes &&
            cgraph->nodes[ಮುಂದಿನ_ಗಂಟು_idx]->op == GGML_OP_RMS_NORM &&
            cgraph->nodes[ಮುಂದಿನ_ಗಂಟು_idx]->src[0] == cgraph->nodes[ಮುಂದಿನ_ಗಂಟು_idx - 1] &&
            ggml_nrows(cgraph->nodes[ಮುಂದಿನ_ಗಂಟು_idx]) == 1 &&
            ctx->device->ಕೂಡಿಸು_rms_fusion) {
            uint32_t size = ggml_vk_rms_partials_ಗಾತ್ರ(ctx, cgraph->nodes[ಗಂಟು_idx]);
            ctx->do_ಕೂಡಿಸು_rms_partials_ಸ್ಥಾನಾಂತರ_calculation = ನಿಜ;
            ಆದರೆ (ctx->prealloc_ಗಾತ್ರ_ಕೂಡಿಸು_rms_partials_ಸ್ಥಾನಾಂತರ + size <= ctx->prealloc_ಗಾತ್ರ_ಕೂಡಿಸು_rms_partials) {
                ctx->do_ಕೂಡಿಸು_rms_partials = ನಿಜ;
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
        ತಾರ್ಕಿಕ need_sync = ಸುಳ್ಳು;

        // Check whether "node" requires synchronization. The node requires synchronization if it
        // overlaps in memory with another unsynchronized node and at least one of them is a write.
        // Destination nodes are checked against both the written/read lists. Source nodes are only
        // checked against the written list. Two nodes overlap in memory if they come from the same
        // buffer and the tensor or view ranges overlap.
        ಸ್ವಯಂಚಾಲಿತ ಸ್ಥಿರಾಂಕ &overlaps_unsynced = [&](ಸ್ಥಿರಾಂಕ ggml_tensor *ಗಂಟು, ಸ್ಥಿರಾಂಕ std::vector<ಸ್ಥಿರಾಂಕ ggml_tensor *> &unsynced_nodes) -> ತಾರ್ಕಿಕ {
            ಆದರೆ (unsynced_nodes.size() == 0) {
                ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
            }
            ಸ್ವಯಂಚಾಲಿತ n_ತಳ = vk_tensor_ಸ್ಥಾನಾಂತರ(ಗಂಟು) + ಗಂಟು->view_offs;
            ಸ್ವಯಂಚಾಲಿತ n_ಗಾತ್ರ = ggml_nbytes(ಗಂಟು);
            ggml_backend_vk_ಬಫರ್_context * a_buf_ctx = (ggml_backend_vk_ಬಫರ್_context *)ಗಂಟು->ಬಫರ್->context;
            vk_ಬಫರ್ a_buf = a_buf_ctx->dev_ಬಫರ್;
            ಗಾಗಿ (ಸ್ವಯಂಚಾಲಿತ &other : unsynced_nodes) {
                ggml_backend_vk_ಬಫರ್_context * o_buf_ctx = (ggml_backend_vk_ಬಫರ್_context *)other->ಬಫರ್->context;
                vk_ಬಫರ್ o_buf = o_buf_ctx->dev_ಬಫರ್;
                ಆದರೆ (a_buf == o_buf) {
                    ಸ್ವಯಂಚಾಲಿತ o_ತಳ = vk_tensor_ಸ್ಥಾನಾಂತರ(other) + other->view_offs;
                    ಸ್ವಯಂಚಾಲಿತ o_ಗಾತ್ರ = ggml_nbytes(other);

                    ಆದರೆ ((o_ತಳ <= n_ತಳ && n_ತಳ < o_ತಳ + o_ಗಾತ್ರ) ||
                        (n_ತಳ <= o_ತಳ && o_ತಳ < n_ತಳ + n_ಗಾತ್ರ)) {
                        ಹಿಂತಿರುಗಿ ನಿಜ;
                    }
                }
            }
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        };

        // For all fused ops, check if the destination node or any of the source
        // nodes require synchronization.
        ಗಾಗಿ (int32_t i = 0; i < ctx->ಅಂಕೆ_additional_fused_ops + 1 && !need_sync; ++i) {
            ಸ್ಥಿರಾಂಕ ggml_tensor *cur_ಗಂಟು = cgraph->nodes[ಗಂಟು_idx + i];
            // If the node actually writes to memory, then check if it needs to sync
            ಆದರೆ (ctx->fused_ops_write_ಮುಖವಾಡ & (1 << i)) {
                ಆದರೆ (overlaps_unsynced(cur_ಗಂಟು, ctx->unsynced_nodes_read) || overlaps_unsynced(cur_ಗಂಟು, ctx->unsynced_nodes_written)) {
                    need_sync = ನಿಜ;
                    ನಿಲ್ಲಿಸು;
                }
            }
            ಗಾಗಿ (uint32_t j = 0; j < GGML_MAX_SRC; ++j) {
                ಆದರೆ (!cur_ಗಂಟು->src[j]) {
                    ಮುಂದುವರಿ;
                }
                ಆದರೆ (overlaps_unsynced(cur_ಗಂಟು->src[j], ctx->unsynced_nodes_written)) {
                    need_sync = ನಿಜ;
                    ನಿಲ್ಲಿಸು;
                }
            }
        }

        ಆದರೆ (need_sync) {
            ಆದರೆ (vk_enable_sync_logger) {
                std::cerr <<  "sync" << std::ಸಾಲು_ಅಂತ್ಯ;
            }
            ctx->unsynced_nodes_written.clear();
            ctx->unsynced_nodes_read.clear();
            ggml_vk_sync_buffers(ctx, compute_ctx);

            ಆದರೆ (vk_perf_logger_enabled && vk_perf_logger_concurrent) {
                ctx->query_ಗಂಟು_idx[ctx->query_idx] = ಗಂಟು_idx;
                compute_ctx->s->ಬಫರ್->buf.writeTimestamp(vk::Pipeline_Stage_Flag_ಬಿಟ್ಟುಗಳು::eAllCommands, ctx->query_pool, ctx->query_idx++);
                ggml_vk_sync_buffers(ctx, compute_ctx);
            }
        }
        // Add all fused nodes to the unsynchronized lists.
        ಗಾಗಿ (int32_t i = 0; i < ctx->ಅಂಕೆ_additional_fused_ops + 1; ++i) {
            ಸ್ಥಿರಾಂಕ ggml_tensor *cur_ಗಂಟು = cgraph->nodes[ಗಂಟು_idx + i];
            // Multiple outputs could be written, e.g. in topk_moe. Add them all to the list.
            ಆದರೆ (ctx->fused_ops_write_ಮುಖವಾಡ & (1 << i)) {
                ctx->unsynced_nodes_written.push_back(cur_ಗಂಟು);
            }
            ಗಾಗಿ (uint32_t j = 0; j < GGML_MAX_SRC; ++j) {
                ಆದರೆ (!cur_ಗಂಟು->src[j]) {
                    ಮುಂದುವರಿ;
                }
                ctx->unsynced_nodes_read.push_back(cur_ಗಂಟು->src[j]);
            }
        }
    }
    ಆದರೆ (vk_enable_sync_logger) {
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < ctx->ಅಂಕೆ_additional_fused_ops + 1; ++i) {
            ಸ್ವಯಂಚಾಲಿತ *n = cgraph->nodes[ಗಂಟು_idx + i];
            std::cerr << ಗಂಟು_idx + i << " " << ggml_op_name(n->op) << " " <<  n->name;
            ಆದರೆ (n->op == GGML_OP_GLU) {
                std::cerr << " " << ggml_glu_op_name(ggml_get_glu_op(n)) << " " << (n->src[1] ? "split" : "single") << " ";
            }
            ಆದರೆ (n->op == GGML_OP_ROPE) {
                ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ mode = ((ಸ್ಥಿರಾಂಕ int32_t *) n->op_params)[2];
                std::cerr << " rope mode: " << mode;
            }
            std::cerr << std::ಸಾಲು_ಅಂತ್ಯ;
        }
    }

    // closed explicitly below, and by the destructor on the early returns
    ggml_vk_debug_label dbg(compute_ctx, cgraph, ಗಂಟು_idx, ctx->ಅಂಕೆ_additional_fused_ops);

    ಆಯ್ಕೆ (ಗಂಟು->op) {
    ಸ್ಥಿತಿ GGML_OP_REPEAT:
        ggml_vk_repeat(ctx, compute_ctx, src0, ಗಂಟು);

        ನಿಲ್ಲಿಸು;
    ಸ್ಥಿತಿ GGML_OP_REPEAT_ಹಿಂಭಾಗ:
        ggml_vk_repeat_ಹಿಂಭಾಗ(ctx, compute_ctx, src0, ಗಂಟು);

        ನಿಲ್ಲಿಸು;
    ಸ್ಥಿತಿ GGML_OP_ACC:
    ಸ್ಥಿತಿ GGML_OP_ಗಣ:
        ggml_vk_acc(ctx, compute_ctx, src0, src1, ಗಂಟು);

        ನಿಲ್ಲಿಸು;
    ಸ್ಥಿತಿ GGML_OP_GET_ಅಡ್ಡಸಾಲುಗಳು:
        ಆದರೆ (ctx->fused_topk_qsa) {
            ggml_vk_topk_qsa(ctx, compute_ctx, cgraph, ಗಂಟು_idx);
        } ಇಲ್ಲದಿದ್ದರೆ {
            ggml_vk_get_ಅಡ್ಡಸಾಲುಗಳು(ctx, compute_ctx, src0, src1, ಗಂಟು);
        }

        ನಿಲ್ಲಿಸು;
    ಸ್ಥಿತಿ GGML_OP_GET_ಅಡ್ಡಸಾಲುಗಳು_ಹಿಂಭಾಗ:
        ggml_vk_get_ಅಡ್ಡಸಾಲುಗಳು_ಹಿಂಭಾಗ(ctx, compute_ctx, src0, src1, ಗಂಟು);

        ನಿಲ್ಲಿಸು;
    ಸ್ಥಿತಿ GGML_OP_ಕೂಡಿಸು:
        ಆದರೆ (ctx->ಅಂಕೆ_additional_fused_ops) {
            ggml_vk_multi_ಕೂಡಿಸು(ctx, compute_ctx, cgraph, ಗಂಟು_idx);
        } ಇಲ್ಲದಿದ್ದರೆ {
            ggml_vk_ಕೂಡಿಸು(ctx, compute_ctx, src0, src1, ಗಂಟು);
        }
        ನಿಲ್ಲಿಸು;
    ಸ್ಥಿತಿ GGML_OP_ಹೊರಹೋಗು_PROD:
        ggml_vk_ಹೊರಹೋಗು_prod(ctx, compute_ctx, src0, src1, ಗಂಟು);
        ನಿಲ್ಲಿಸು;
    ಸ್ಥಿತಿ GGML_OP_SUB:
        ggml_vk_sub(ctx, compute_ctx, src0, src1, ಗಂಟು);

        ನಿಲ್ಲಿಸು;
    ಸ್ಥಿತಿ GGML_OP_MUL:
        ಆದರೆ (ctx->ಅಂಕೆ_additional_fused_ops) {
            ggml_vk_snake_dispatch_fused(ctx, compute_ctx, cgraph, ಗಂಟು_idx);
        } ಇಲ್ಲದಿದ್ದರೆ {
            ggml_vk_mul(ctx, compute_ctx, src0, src1, ಗಂಟು);
        }

        ನಿಲ್ಲಿಸು;
    ಸ್ಥಿತಿ GGML_OP_DIV:
        ggml_vk_div(ctx, compute_ctx, src0, src1, ಗಂಟು);

        ನಿಲ್ಲಿಸು;
    ಸ್ಥಿತಿ GGML_OP_ಕೂಡಿಸು_ID:
        ggml_vk_ಕೂಡಿಸು_id(ctx, compute_ctx, src0, src1, src2, ಗಂಟು);

        ನಿಲ್ಲಿಸು;
    ಸ್ಥಿತಿ GGML_OP_CONCAT:
        ggml_vk_concat(ctx, compute_ctx, src0, src1, ಗಂಟು);

        ನಿಲ್ಲಿಸು;
    ಸ್ಥಿತಿ GGML_OP_UPSCALE:
        ggml_vk_upscale(ctx, compute_ctx, src0, ಗಂಟು);

        ನಿಲ್ಲಿಸು;
    ಸ್ಥಿತಿ GGML_OP_ADD1:
        ggml_vk_add1(ctx, compute_ctx, src0, src1, ಗಂಟು);

        ನಿಲ್ಲಿಸು;
    ಸ್ಥಿತಿ GGML_OP_ARANGE:
        ggml_vk_arange(ctx, compute_ctx, ಗಂಟು);

        ನಿಲ್ಲಿಸು;
    ಸ್ಥಿತಿ GGML_OP_FILL:
        ggml_vk_fill(ctx, compute_ctx, ಗಂಟು);

        ನಿಲ್ಲಿಸು;
    ಸ್ಥಿತಿ GGML_OP_SCALE:
        ಆದರೆ (ctx->fused_hc_post_gate) {
            ggml_tensor * hc_post = cgraph->nodes[ಗಂಟು_idx + ctx->ಅಂಕೆ_additional_fused_ops];
            ggml_vk_dsv4_hc_post(ctx, compute_ctx, hc_post->src[0], hc_post->src[1], hc_post->src[2], hc_post->src[3], hc_post, ಗಂಟು);
        } ಇಲ್ಲದಿದ್ದರೆ {
            ggml_vk_scale(ctx, compute_ctx, src0, ಗಂಟು);
        }

        ನಿಲ್ಲಿಸು;
    ಸ್ಥಿತಿ GGML_OP_SQR:
        ggml_vk_sqr(ctx, compute_ctx, src0, ಗಂಟು);

        ನಿಲ್ಲಿಸು;
    ಸ್ಥಿತಿ GGML_OP_SQRT:
        ggml_vk_sqrt(ctx, compute_ctx, src0, ಗಂಟು);

        ನಿಲ್ಲಿಸು;
    ಸ್ಥಿತಿ GGML_OP_SIN:
        ggml_vk_sin(ctx, compute_ctx, src0, ಗಂಟು);

        ನಿಲ್ಲಿಸು;
    ಸ್ಥಿತಿ GGML_OP_COS:
        ggml_vk_cos(ctx, compute_ctx, src0, ಗಂಟು);

        ನಿಲ್ಲಿಸು;
    ಸ್ಥಿತಿ GGML_OP_LOG:
        ggml_vk_log(ctx, compute_ctx, src0, ಗಂಟು);

        ನಿಲ್ಲಿಸು;
    ಸ್ಥಿತಿ GGML_OP_TRI:
        ggml_vk_tri(ctx, compute_ctx, src0, ಗಂಟು);

        ನಿಲ್ಲಿಸು;
    ಸ್ಥಿತಿ GGML_OP_DIAG:
        ggml_vk_diag(ctx, compute_ctx, src0, ಗಂಟು);

        ನಿಲ್ಲಿಸು;
    ಸ್ಥಿತಿ GGML_OP_CLAMP:
        ggml_vk_clamp(ctx, compute_ctx, src0, ಗಂಟು);

        ನಿಲ್ಲಿಸು;
    ಸ್ಥಿತಿ GGML_OP_PAD:
        ggml_vk_pad(ctx, compute_ctx, src0, ಗಂಟು);

        ನಿಲ್ಲಿಸು;
    ಸ್ಥಿತಿ GGML_OP_PAD_REFLECT_1D:
        ggml_vk_pad_reflect_1d(ctx, compute_ctx, src0, ಗಂಟು);

        ನಿಲ್ಲಿಸು;
    ಸ್ಥಿತಿ GGML_OP_ROLL:
        ggml_vk_roll(ctx, compute_ctx, src0, ಗಂಟು);

        ನಿಲ್ಲಿಸು;
    ಸ್ಥಿತಿ GGML_OP_CPY:
    ಸ್ಥಿತಿ GGML_OP_CONT:
    ಸ್ಥಿತಿ GGML_OP_DUP:
        ggml_vk_cpy(ctx, compute_ctx, src0, ಗಂಟು);

        ನಿಲ್ಲಿಸು;
    ಸ್ಥಿತಿ GGML_OP_ಗಣ_ಅಡ್ಡಸಾಲುಗಳು:
        ggml_vk_ಗಣ_ಅಡ್ಡಸಾಲುಗಳು(ctx, compute_ctx, src0, src1, ಗಂಟು);

        ನಿಲ್ಲಿಸು;
    ಸ್ಥಿತಿ GGML_OP_SILU_ಹಿಂಭಾಗ:
        ggml_vk_silu_ಹಿಂಭಾಗ(ctx, compute_ctx, src0, src1, ಗಂಟು);

        ನಿಲ್ಲಿಸು;
    ಸ್ಥಿತಿ GGML_OP_NORM:
        ggml_vk_norm(ctx, compute_ctx, src0, ಗಂಟು);

        ನಿಲ್ಲಿಸು;
    ಸ್ಥಿತಿ GGML_OP_ಗುಂಪು_NORM:
        ggml_vk_ಗುಂಪು_norm(ctx, compute_ctx, src0, ಗಂಟು);

        ನಿಲ್ಲಿಸು;
    ಸ್ಥಿತಿ GGML_OP_RMS_NORM:
        ggml_vk_rms_norm(ctx, compute_ctx, cgraph, ಗಂಟು_idx, (ದಶಮಾಂಶ *)ಗಂಟು->op_params);
        ನಿಲ್ಲಿಸು;
    ಸ್ಥಿತಿ GGML_OP_RMS_NORM_ಹಿಂಭಾಗ:
        ggml_vk_rms_norm_ಹಿಂಭಾಗ(ctx, compute_ctx, src0, src1, ಗಂಟು);

        ನಿಲ್ಲಿಸು;
    ಸ್ಥಿತಿ GGML_OP_L2_NORM:
        ggml_vk_l2_norm(ctx, compute_ctx, src0, ಗಂಟು);

        ನಿಲ್ಲಿಸು;
    ಸ್ಥಿತಿ GGML_OP_UNARY:
        ಆದರೆ (ctx->fused_topk_moe_ಬಹುಲಕ != TOPK_MOE_ಎಣಿಕೆ) {
            ggml_vk_topk_moe(ctx, compute_ctx, cgraph, ಗಂಟು_idx);
            ನಿಲ್ಲಿಸು;
        }
        ಆದರೆ (ctx->ಅಂಕೆ_additional_fused_ops) {
            ggml_vk_unary_mul(ctx, compute_ctx, cgraph, ಗಂಟು_idx);
            ನಿಲ್ಲಿಸು;
        }

        ಆಯ್ಕೆ (ggml_get_unary_op(ಗಂಟು)) {
        ಸ್ಥಿತಿ GGML_UNARY_OP_ELU:
        ಸ್ಥಿತಿ GGML_UNARY_OP_EXP:
        ಸ್ಥಿತಿ GGML_UNARY_OP_EXPM1:
        ಸ್ಥಿತಿ GGML_UNARY_OP_SILU:
        ಸ್ಥಿತಿ GGML_UNARY_OP_GELU:
        ಸ್ಥಿತಿ GGML_UNARY_OP_GELU_ERF:
        ಸ್ಥಿತಿ GGML_UNARY_OP_GELU_ಶೀಘ್ರ:
        ಸ್ಥಿತಿ GGML_UNARY_OP_RELU:
        ಸ್ಥಿತಿ GGML_UNARY_OP_NEG:
        ಸ್ಥಿತಿ GGML_UNARY_OP_TANH:
        ಸ್ಥಿತಿ GGML_UNARY_OP_SIGMOID:
        ಸ್ಥಿತಿ GGML_UNARY_OP_HARDSIGMOID:
        ಸ್ಥಿತಿ GGML_UNARY_OP_HARDSWISH:
        ಸ್ಥಿತಿ GGML_UNARY_OP_ABS:
        ಸ್ಥಿತಿ GGML_UNARY_OP_SOFTPLUS:
        ಸ್ಥಿತಿ GGML_UNARY_OP_ಹೆಜ್ಜೆ:
        ಸ್ಥಿತಿ GGML_UNARY_OP_ಸುತ್ತು:
        ಸ್ಥಿತಿ GGML_UNARY_OP_CEIL:
        ಸ್ಥಿತಿ GGML_UNARY_OP_FLOOR:
        ಸ್ಥಿತಿ GGML_UNARY_OP_TRUNC:
        ಸ್ಥಿತಿ GGML_UNARY_OP_SGN:
            ggml_vk_unary(ctx, compute_ctx, src0, ಗಂಟು);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_UNARY_OP_XIELU:
            ggml_vk_xielu(ctx, compute_ctx, src0, ಗಂಟು);
            ನಿಲ್ಲಿಸು;
        ಪೂರ್ವನಿಯೋಜಿತ:
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }
        ನಿಲ್ಲಿಸು;
    ಸ್ಥಿತಿ GGML_OP_GLU:
        ಆಯ್ಕೆ (ggml_get_glu_op(ಗಂಟು)) {
        ಸ್ಥಿತಿ GGML_GLU_OP_GEGLU:
        ಸ್ಥಿತಿ GGML_GLU_OP_REGLU:
        ಸ್ಥಿತಿ GGML_GLU_OP_SWIGLU:
        ಸ್ಥಿತಿ GGML_GLU_OP_SWIGLU_OAI:
        ಸ್ಥಿತಿ GGML_GLU_OP_GEGLU_ERF:
        ಸ್ಥಿತಿ GGML_GLU_OP_GEGLU_ಶೀಘ್ರ:
        ಸ್ಥಿತಿ GGML_GLU_OP_SWIGLU_CLAMP:
            ggml_vk_glu(ctx, compute_ctx, src0, src1, ಗಂಟು);
            ನಿಲ್ಲಿಸು;
        ಪೂರ್ವನಿಯೋಜಿತ:
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }
        ನಿಲ್ಲಿಸು;
    ಸ್ಥಿತಿ GGML_OP_DIAG_ಮುಖವಾಡ_INF:
        ggml_vk_diag_ಮುಖವಾಡ_inf(ctx, compute_ctx, src0, ಗಂಟು);

        ನಿಲ್ಲಿಸು;
    ಸ್ಥಿತಿ GGML_OP_SOFT_MAX:
        ಆದರೆ (ctx->fused_topk_moe_ಬಹುಲಕ != TOPK_MOE_ಎಣಿಕೆ) {
            ggml_vk_topk_moe(ctx, compute_ctx, cgraph, ಗಂಟು_idx);
        } ಇಲ್ಲದಿದ್ದರೆ {
            ggml_vk_soft_max(ctx, compute_ctx, src0, src1, src2, ಗಂಟು);
        }

        ನಿಲ್ಲಿಸು;
    ಸ್ಥಿತಿ GGML_OP_SOFT_MAX_ಹಿಂಭಾಗ:
        ggml_vk_soft_max_ಹಿಂಭಾಗ(ctx, compute_ctx, src0, src1, ಗಂಟು);

        ನಿಲ್ಲಿಸು;
    ಸ್ಥಿತಿ GGML_OP_ROPE:
        ggml_vk_rope(ctx, compute_ctx, cgraph, ಗಂಟು_idx, ಸುಳ್ಳು);

        ನಿಲ್ಲಿಸು;
    ಸ್ಥಿತಿ GGML_OP_ROPE_ಹಿಂಭಾಗ:
        ggml_vk_rope(ctx, compute_ctx, cgraph, ಗಂಟು_idx, ನಿಜ);

        ನಿಲ್ಲಿಸು;
    ಸ್ಥಿತಿ GGML_OP_ARGSORT:
        ಆದರೆ (ctx->fused_topk_moe_ಬಹುಲಕ != TOPK_MOE_ಎಣಿಕೆ) {
            ggml_vk_topk_moe(ctx, compute_ctx, cgraph, ಗಂಟು_idx);
        } ಇಲ್ಲದಿದ್ದರೆ {
            ggml_vk_argsort(ctx, compute_ctx, src0, ಗಂಟು);
        }

        ನಿಲ್ಲಿಸು;
    ಸ್ಥಿತಿ GGML_OP_ತುದಿ_K:
        ggml_vk_topk(ctx, compute_ctx, src0, ಗಂಟು);

        ನಿಲ್ಲಿಸು;
    ಸ್ಥಿತಿ GGML_OP_ಮೊತ್ತ:
        ggml_vk_ಮೊತ್ತ(ctx, compute_ctx, src0, ಗಂಟು);

        ನಿಲ್ಲಿಸು;
    ಸ್ಥಿತಿ GGML_OP_ಮೊತ್ತ_ಅಡ್ಡಸಾಲುಗಳು:
        ggml_vk_ಮೊತ್ತ_ಅಡ್ಡಸಾಲುಗಳು(ctx, compute_ctx, src0, ಗಂಟು);

        ನಿಲ್ಲಿಸು;
    ಸ್ಥಿತಿ GGML_OP_CUMSUM:
        ggml_vk_cumsum(ctx, compute_ctx, src0, ಗಂಟು);

        ನಿಲ್ಲಿಸು;
    ಸ್ಥಿತಿ GGML_OP_DSV4_HC_COMB:
        ggml_vk_dsv4_hc_comb(ctx, compute_ctx, src0, src1, src2, ಗಂಟು);

        ನಿಲ್ಲಿಸು;
    ಸ್ಥಿತಿ GGML_OP_DSV4_HC_PRE:
        ggml_vk_dsv4_hc_pre(ctx, compute_ctx, src0, src1, ಗಂಟು);

        ನಿಲ್ಲಿಸು;
    ಸ್ಥಿತಿ GGML_OP_DSV4_HC_POST:
        ggml_vk_dsv4_hc_post(ctx, compute_ctx, src0, src1, src2, src3, ಗಂಟು);

        ನಿಲ್ಲಿಸು;
    ಸ್ಥಿತಿ GGML_OP_ಸರಾಸರಿ:
        ggml_vk_ಸರಾಸರಿ(ctx, compute_ctx, src0, ಗಂಟು);

        ನಿಲ್ಲಿಸು;
    ಸ್ಥಿತಿ GGML_OP_ARGMAX:
        ggml_vk_argmax(ctx, compute_ctx, src0, ಗಂಟು);

        ನಿಲ್ಲಿಸು;
    ಸ್ಥಿತಿ GGML_OP_CROSS_ENTROPY_LOSS:
        ggml_vk_cross_entropy_loss(ctx, compute_ctx, ಗಂಟು);

        ನಿಲ್ಲಿಸು;
    ಸ್ಥಿತಿ GGML_OP_CROSS_ENTROPY_LOSS_ಹಿಂಭಾಗ:
        ggml_vk_cross_entropy_loss_ಹಿಂಭಾಗ(ctx, compute_ctx, ಗಂಟು);

        ನಿಲ್ಲಿಸು;
    ಸ್ಥಿತಿ GGML_OP_ಎಣಿಕೆ_EQUAL:
        ggml_vk_ಎಣಿಕೆ_equal(ctx, compute_ctx, src0, src1, ಗಂಟು);

        ನಿಲ್ಲಿಸು;
    ಸ್ಥಿತಿ GGML_OP_SOLVE_TRI:
        ggml_vk_solve_tri(ctx, compute_ctx, src0, src1, ಗಂಟು);

        ನಿಲ್ಲಿಸು;
    ಸ್ಥಿತಿ GGML_OP_IM2_ಲಂಬಸಾಲು:
        ggml_vk_im2col(ctx, compute_ctx, src0, src1, ಗಂಟು);

        ನಿಲ್ಲಿಸು;
    ಸ್ಥಿತಿ GGML_OP_IM2_ಲಂಬಸಾಲು_3_D:
        ggml_vk_im2col_3d(ctx, compute_ctx, src0, src1, ಗಂಟು);

        ನಿಲ್ಲಿಸು;
    ಸ್ಥಿತಿ GGML_OP_TIMESTEP_EMBEDDING:
        ggml_vk_timestep_embedding(ctx, compute_ctx, src0, ಗಂಟು);

        ನಿಲ್ಲಿಸು;
    ಸ್ಥಿತಿ GGML_OP_COL2IM_1D:
        ggml_vk_col2im_1d(ctx, compute_ctx, src0, ಗಂಟು);

        ನಿಲ್ಲಿಸು;
    ಸ್ಥಿತಿ GGML_OP_CONV_TRANSPOSE_1D:
        ggml_vk_conv_transpose_1d(ctx, compute_ctx, src0, src1, ಗಂಟು);

        ನಿಲ್ಲಿಸು;
    ಸ್ಥಿತಿ GGML_OP_POOL_1D:
        ggml_vk_pool_1d(ctx, compute_ctx, src0, ಗಂಟು);

        ನಿಲ್ಲಿಸು;
    ಸ್ಥಿತಿ GGML_OP_POOL_2D:
        ggml_vk_pool_2d(ctx, compute_ctx, src0, ಗಂಟು);

        ನಿಲ್ಲಿಸು;
    ಸ್ಥಿತಿ GGML_OP_CONV_2D:
    ಸ್ಥಿತಿ GGML_OP_CONV_TRANSPOSE_2D:
        ggml_vk_conv_2d(ctx, compute_ctx, src0, src1, ಗಂಟು);

        ನಿಲ್ಲಿಸು;
    ಸ್ಥಿತಿ GGML_OP_CONV_3D:
        ggml_vk_conv_3d(ctx, compute_ctx, src0, src1, ಗಂಟು);

        ನಿಲ್ಲಿಸು;
    ಸ್ಥಿತಿ GGML_OP_CONV_2D_DW:
        ggml_vk_conv_2d_dw(ctx, compute_ctx, src0, src1, ಗಂಟು);

        ನಿಲ್ಲಿಸು;
    ಸ್ಥಿತಿ GGML_OP_LEAKY_RELU:
        ggml_vk_leaky_relu(ctx, compute_ctx, src0, ಗಂಟು);

        ನಿಲ್ಲಿಸು;
    ಸ್ಥಿತಿ GGML_OP_MUL_MAT:
        ggml_vk_mul_mat(ctx, compute_ctx, cgraph, ಗಂಟು_idx);

        ನಿಲ್ಲಿಸು;
    ಸ್ಥಿತಿ GGML_OP_MUL_MAT_ID:
        ggml_vk_mul_mat_id(ctx, compute_ctx, cgraph, ಗಂಟು_idx);

        ನಿಲ್ಲಿಸು;

    ಸ್ಥಿತಿ GGML_OP_FLASH_ATTN_EXT:
        ggml_vk_flash_attn(ctx, compute_ctx, src0, src1, src2, src3, ಗಂಟು->src[4], ಗಂಟು);

        ನಿಲ್ಲಿಸು;

    ಸ್ಥಿತಿ GGML_OP_RWKV_WKV6:
        ggml_vk_rwkv_wkv6(ctx, compute_ctx, ಗಂಟು);

        ನಿಲ್ಲಿಸು;

    ಸ್ಥಿತಿ GGML_OP_RWKV_WKV7:
        ggml_vk_rwkv_wkv7(ctx, compute_ctx, ಗಂಟು);

        ನಿಲ್ಲಿಸು;

    ಸ್ಥಿತಿ GGML_OP_GATED_ರೇಖೀಯ_ATTN:
        ggml_vk_gated_ರೇಖೀಯ_attn(ctx, compute_ctx, ಗಂಟು);

        ನಿಲ್ಲಿಸು;

    ಸ್ಥಿತಿ GGML_OP_LIGHTNING_INDEXER:
        ggml_vk_lightning_indexer(ctx, compute_ctx, ಗಂಟು);

        ನಿಲ್ಲಿಸು;

    ಸ್ಥಿತಿ GGML_OP_GATED_DELTA_NET:
        ggml_vk_gated_delta_net(ctx, compute_ctx, ಗಂಟು);

        ನಿಲ್ಲಿಸು;

    ಸ್ಥಿತಿ GGML_OP_SSM_ಪರಿಶೀಲನೆ:
        ggml_vk_ssm_ಪರಿಶೀಲನೆ(ctx, compute_ctx, ಗಂಟು);

        ನಿಲ್ಲಿಸು;

    ಸ್ಥಿತಿ GGML_OP_SSM_CONV:
        ggml_vk_ssm_conv(ctx, compute_ctx, cgraph, ಗಂಟು_idx);

        ನಿಲ್ಲಿಸು;

    ಸ್ಥಿತಿ GGML_OP_OPT_ಹೆಜ್ಜೆ_ADAMW:
        ggml_vk_opt_ಹೆಜ್ಜೆ_adamw(ctx, compute_ctx, ಗಂಟು);

        ನಿಲ್ಲಿಸು;

    ಸ್ಥಿತಿ GGML_OP_OPT_ಹೆಜ್ಜೆ_SGD:
        ggml_vk_opt_ಹೆಜ್ಜೆ_sgd(ctx, compute_ctx, src0, src1, src2, ಗಂಟು);

        ನಿಲ್ಲಿಸು;
    ಪೂರ್ವನಿಯೋಜಿತ:
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    // the submit path below can end the command buffer, so close the region first
    dbg.close();

    ctx->tensor_ctxs[ಗಂಟು_idx] = compute_ctx;

#ಆದರೆ defined(GGML_VULKAN_CHECK_RESULTS)
    // Force context reset on each node so that each tensor ends up in its own context
    // and can be run and compared to its CPU equivalent separately
    ಕೊನೆಯ_ಗಂಟು = ನಿಜ;
#ಆದರೆ_ಅಂತ್ಯ

    ಆದರೆ (submit || ಕೊನೆಯ_ಗಂಟು) {
        ggml_vk_ctx_ಕೊನೆ(compute_ctx);

        // TODO probably it'd be better to pass a exit_node flag to ggml_vk_compute_forward
        ಆದರೆ (ಕೊನೆಯ_ಗಂಟು) {
            compute_ctx->exit_tensor_idx = ಗಂಟು_idx_begin;
        }
        ಇಲ್ಲದಿದ್ದರೆ {
            compute_ctx->exit_tensor_idx = -1;
        }

        ctx->compute_ctx.reset();

        ggml_vk_compute_forward(ctx, cgraph, ಗಂಟು_begin, ಗಂಟು_idx_begin, almost_ready);
    }
    ಹಿಂತಿರುಗಿ ನಿಜ;
}

ಶೂನ್ಯ ggml_vk_compute_forward(ggml_backend_vk_context * ctx, ggml_cgraph * cgraph, ggml_tensor * tensor, ಪೂರ್ಣಾಂಕ tensor_idx, ತಾರ್ಕಿಕ almost_ready) {
    GGML_ಬಳಸದ(cgraph);
    GGML_ಬಳಸದ(tensor);

    VK_LOG_DEBUG("ggml_vk_compute_forward(" << tensor << ", name=" << tensor->name << ", op=" << ggml_op_name(tensor->op) << ", type=" << tensor->type << ", ne0=" << tensor->ne[0] << ", ne1=" << tensor->ne[1] << ", ne2=" << tensor->ne[2] << ", ne3=" << tensor->ne[3] << ", nb0=" << tensor->nb[0] << ", nb1=" << tensor->nb[1] << ", nb2=" << tensor->nb[2] << ", nb3=" << tensor->nb[3] << ", view_src=" << tensor->view_src << ", view_offs=" << tensor->view_offs << ")");

    vk_context subctx = ctx->tensor_ctxs[tensor_idx].ಬೀಗ();

    // Only run if ctx hasn't been submitted yet
    ಆದರೆ (!subctx->seqs.ಖಾಲಿಗಳು()) {
#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_VULKAN_CHECK_RESULTS
        ggml_vk_check_results_0(ctx, cgraph, tensor_idx);
#ಆದರೆ_ಅಂತ್ಯ

        // Do staging buffer copies
        ಗಾಗಿ (ಸ್ವಯಂಚಾಲಿತ& cpy : subctx->ಒಳ_memcpys) {
            ಸ್ಮರಣೆ_ನಕಲು(cpy.dst, cpy.src, cpy.n);
        }

        ಗಾಗಿ (ಸ್ವಯಂಚಾಲಿತ& mset : subctx->memsets) {
            memset(mset.dst, mset.val, mset.n);
        }

        ಆದರೆ (ctx->device->serialize_submissions) {
            ggml_vk_submit(subctx, ctx->fence);
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (almost_ready && !ctx->almost_ready_fence_pending) {
            ggml_vk_submit(subctx, ctx->almost_ready_fence);
            ctx->almost_ready_fence_pending = ನಿಜ;
        } ಇಲ್ಲದಿದ್ದರೆ {
            ggml_vk_submit(subctx, {});
        }
        ctx->submit_pending = ನಿಜ;

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_VULKAN_CHECK_RESULTS
        ggml_vk_synchronize(ctx);
        ggml_vk_check_results_1(ctx, cgraph, tensor_idx);
#ಆದರೆ_ಅಂತ್ಯ
    }

    ಆದರೆ (tensor_idx == subctx->exit_tensor_idx) {
        // Do staging buffer copies
        ಗಾಗಿ (ಸ್ವಯಂಚಾಲಿತ& cpy : subctx->ಹೊರಹೋಗು_memcpys) {
            ಸ್ಮರಣೆ_ನಕಲು(cpy.dst, cpy.src, cpy.n);
        }
        subctx->ಒಳ_memcpys.clear();
        subctx->ಹೊರಹೋಗು_memcpys.clear();
        subctx->memsets.clear();
    }
}

ಶೂನ್ಯ ggml_vk_ನಕ್ಷೆ_cleanup(ggml_backend_vk_context * ctx) {
    VK_LOG_DEBUG("ggml_vk_graph_cleanup()");
    ctx->prealloc_y_ಕೊನೆಯ_pipeline_ಬಳಸಿದ = {};
    ctx->prealloc_y_ಕೊನೆಯ_tensor_ಬಳಸಿದ = ಶೂನ್ಯಸೂಚಕ;
    ctx->prealloc_y_ಕೊನೆಯ_k_padded = ಸುಳ್ಳು;

    ctx->unsynced_nodes_written.clear();
    ctx->unsynced_nodes_read.clear();
    ctx->prealloc_x_need_sync = ctx->prealloc_y_need_sync = ctx->prealloc_ವಿಭಜಿಸು_k_need_sync = ಸುಳ್ಳು;

    ggml_vk_command_pool_cleanup(ctx->device, ctx->compute_cmd_pool);
    ಆದರೆ (ctx->device->async_use_transfer_ಸರತಿ) {
        ggml_vk_command_pool_cleanup(ctx->device, ctx->transfer_cmd_pool);
    }

    ಗಾಗಿ (size_t i = 0; i < ctx->gc.semaphores.size(); i++) {
        ctx->device->device.destroySemaphore({ ctx->gc.semaphores[i].s });
    }
    ctx->gc.semaphores.clear();

    ಗಾಗಿ (size_t i = 0; i < ctx->gc.tl_semaphores.size(); i++) {
        ctx->device->device.destroySemaphore({ ctx->gc.tl_semaphores[i].s });
    }
    ctx->gc.tl_semaphores.clear();
    ctx->semaphore_idx = 0;

    ctx->event_idx = 0;

    ಗಾಗಿ (ಸ್ವಯಂಚಾಲಿತ& event : ctx->gc.events) {
        ctx->device->device.resetEvent(event);
    }

    ctx->tensor_ctxs.clear();
    ctx->gc.contexts.clear();
    ctx->pipeline_descriptor_ಗಣ_requirements = 0;
    ctx->descriptor_ಗಣ_idx = 0;
}

ಶೂನ್ಯ ggml_vk_cleanup(ggml_backend_vk_context * ctx) {
    VK_LOG_DEBUG("ggml_vk_cleanup(" << ctx->name << ")");
    // discard any unsubmitted command buffers
    ctx->compute_ctx.reset();
    // wait for any pending command buffers to finish
    ggml_vk_synchronize(ctx);

    ggml_vk_ನಕ್ಷೆ_cleanup(ctx);

    ggml_vk_destroy_ಬಫರ್(ctx->prealloc_x);
    ggml_vk_destroy_ಬಫರ್(ctx->prealloc_y);
    ggml_vk_destroy_ಬಫರ್(ctx->prealloc_ವಿಭಜಿಸು_k);
    ggml_vk_destroy_ಬಫರ್(ctx->prealloc_ಕೂಡಿಸು_rms_partials);
    ggml_vk_destroy_ಬಫರ್(ctx->sync_staging);

    ctx->prealloc_y_ಕೊನೆಯ_pipeline_ಬಳಸಿದ = ಶೂನ್ಯಸೂಚಕ;
    ctx->prealloc_y_ಕೊನೆಯ_tensor_ಬಳಸಿದ = ಶೂನ್ಯಸೂಚಕ;
    ctx->prealloc_y_ಕೊನೆಯ_k_padded = ಸುಳ್ಳು;

    ctx->prealloc_ಗಾತ್ರ_x = 0;
    ctx->prealloc_ಗಾತ್ರ_y = 0;
    ctx->prealloc_ಗಾತ್ರ_ವಿಭಜಿಸು_k = 0;

    ಗಾಗಿ (ಸ್ವಯಂಚಾಲಿತ& event : ctx->gc.events) {
        ctx->device->device.destroyEvent(event);
    }
    ctx->gc.events.clear();

    ctx->device->device.destroyFence(ctx->fence);
    ctx->device->device.destroyFence(ctx->almost_ready_fence);

    ಗಾಗಿ (ಸ್ವಯಂಚಾಲಿತ& pool : ctx->descriptor_pools) {
        ctx->device->device.destroyDescriptorPool(pool);
    }
    ctx->descriptor_pools.clear();
    ctx->descriptor_sets.clear();
    ctx->descriptor_ಗಣ_bindings.clear();

    ctx->compute_cmd_pool.destroy(ctx->device->device);
    ಆದರೆ (ctx->device->async_use_transfer_ಸರತಿ) {
        ctx->device->device.destroySemaphore(ctx->transfer_semaphore.s);

        ctx->transfer_cmd_pool.destroy(ctx->device->device);
    }
    ಆದರೆ (vk_perf_logger_enabled) {
        ctx->perf_logger->print_timings(ನಿಜ);
    }
}

ಪೂರ್ಣಾಂಕ ggml_vk_get_device_ಎಣಿಕೆ() {
    ggml_vk_instance_init();

    ಹಿಂತಿರುಗಿ vk_instance.device_indices.size();
}

ಶೂನ್ಯ ggml_vk_get_device_description(ಪೂರ್ಣಾಂಕ device, ಅಕ್ಷರ * description, size_t description_ಗಾತ್ರ) {
    ggml_vk_instance_init();

    std::vector<vk::PhysicalDevice> devices = vk_instance.instance.enumeratePhysicalDevices();

    vk::PhysicalDeviceProperties props;
    devices[device].getProperties(&props);

    snprintf(description, description_ಗಾತ್ರ, "%s", props.deviceName.ದತ್ತಾಂಶ());
}

ತಾರ್ಕಿಕ ggml_backend_ಬಫರ್_is_vk(ggml_backend_ಬಫರ್_t ಬಫರ್) {
    ಹಿಂತಿರುಗಿ ಬಫರ್->buft->iface.get_name == ggml_backend_vk_ಬಫರ್_type_name;
}

ಶೂನ್ಯ ggml_backend_vk_ಬಫರ್_free_ಬಫರ್(ggml_backend_ಬಫರ್_t ಬಫರ್) {
    VK_LOG_ಸ್ಮರಣೆ("ggml_backend_vk_buffer_free_buffer()");
    ggml_backend_vk_ಬಫರ್_context * ctx = (ggml_backend_vk_ಬಫರ್_context *)ಬಫರ್->context;
    ggml_vk_destroy_ಬಫರ್(ctx->dev_ಬಫರ್);
    ಅಳಿಸು ctx;
}

ಶೂನ್ಯ * ggml_backend_vk_ಬಫರ್_get_ತಳ(ggml_backend_ಬಫರ್_t ಬಫರ್) {
    ಹಿಂತಿರುಗಿ vk_ptr_ತಳ;

    ಬಳಸದ(ಬಫರ್);
}

ಎಣಿಕೆ ggml_ಸ್ಥಿತಿ ggml_backend_vk_ಬಫರ್_init_tensor(ggml_backend_ಬಫರ್_t ಬಫರ್, ggml_tensor * tensor) {
    VK_LOG_DEBUG("ggml_backend_vk_buffer_init_tensor(" << ಬಫರ್ << " (" << ಬಫರ್->context << "), " << tensor << ")");
    ಆದರೆ (tensor->view_src != ಶೂನ್ಯಸೂಚಕ) {
        GGML_ASSERT(tensor->view_src->ಬಫರ್->buft == ಬಫರ್->buft);
    }
    ಹಿಂತಿರುಗಿ GGML_ಸ್ಥಿತಿ_SUCCESS;
}

ಶೂನ್ಯ ggml_backend_vk_ಬಫರ್_memset_tensor(ggml_backend_ಬಫರ್_t ಬಫರ್, ggml_tensor * tensor, uint8_t value, size_t ಸ್ಥಾನಾಂತರ, size_t size) {
    VK_LOG_DEBUG("ggml_backend_vk_buffer_memset_tensor(" << ಬಫರ್ << ", " << tensor << ", " << value << ", " << ಸ್ಥಾನಾಂತರ << ", " << size << ")");
    ggml_backend_vk_ಬಫರ್_context * buf_ctx = (ggml_backend_vk_ಬಫರ್_context *)ಬಫರ್->context;
    vk_ಬಫರ್ buf = buf_ctx->dev_ಬಫರ್;

    ಆದರೆ (size == 0) {
        ಹಿಂತಿರುಗಿ;
    }

    uint32_t val32 = (uint32_t)value * 0x01010101;
    ggml_vk_ಬಫರ್_memset(buf, vk_tensor_ಸ್ಥಾನಾಂತರ(tensor) + tensor->view_offs + ಸ್ಥಾನಾಂತರ, val32, size);
}

ಶೂನ್ಯ ggml_backend_vk_ಬಫರ್_ಗಣ_tensor(ggml_backend_ಬಫರ್_t ಬಫರ್, ggml_tensor * tensor, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ದತ್ತಾಂಶ, size_t ಸ್ಥಾನಾಂತರ, size_t size) {
    VK_LOG_DEBUG("ggml_backend_vk_buffer_set_tensor(" << ಬಫರ್ << ", " << tensor << ", " << ದತ್ತಾಂಶ << ", " << ಸ್ಥಾನಾಂತರ << ", " << size << ")");
    ggml_backend_vk_ಬಫರ್_context * buf_ctx = (ggml_backend_vk_ಬಫರ್_context *)ಬಫರ್->context;
    vk_ಬಫರ್ buf = buf_ctx->dev_ಬಫರ್;

    ಆದರೆ (size == 0) {
        ಹಿಂತಿರುಗಿ;
    }

    ggml_vk_ಬಫರ್_write(buf, vk_tensor_ಸ್ಥಾನಾಂತರ(tensor) + tensor->view_offs + ಸ್ಥಾನಾಂತರ, ದತ್ತಾಂಶ, size);
}

ಶೂನ್ಯ ggml_backend_vk_ಬಫರ್_ಗಣ_tensor_2d(ggml_backend_ಬಫರ್_t ಬಫರ್, ggml_tensor * tensor, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ದತ್ತಾಂಶ, size_t ಸ್ಥಾನಾಂತರ,
                                                 size_t size, size_t n_copies, size_t stride_tensor, size_t stride_ದತ್ತಾಂಶ) {
    VK_LOG_DEBUG("ggml_backend_vk_buffer_set_tensor_2d(" << ಬಫರ್ << ", " << tensor << ", " << ದತ್ತಾಂಶ << ", " << ಸ್ಥಾನಾಂತರ << ", " << size << ", " <<
                 n_copies << ", " << stride_tensor << ", " << stride_ದತ್ತಾಂಶ << ")");
    ggml_backend_vk_ಬಫರ್_context * buf_ctx = (ggml_backend_vk_ಬಫರ್_context *)ಬಫರ್->context;
    vk_ಬಫರ್ buf = buf_ctx->dev_ಬಫರ್;

    ಆದರೆ (size == 0) {
        ಹಿಂತಿರುಗಿ;
    }

    ggml_vk_ಬಫರ್_write_2d(buf, vk_tensor_ಸ್ಥಾನಾಂತರ(tensor) + tensor->view_offs + ಸ್ಥಾನಾಂತರ, ದತ್ತಾಂಶ, stride_ದತ್ತಾಂಶ, stride_tensor, size, n_copies);
}

ಶೂನ್ಯ ggml_backend_vk_ಬಫರ್_get_tensor(ggml_backend_ಬಫರ್_t ಬಫರ್, ಸ್ಥಿರಾಂಕ ggml_tensor * tensor, ಶೂನ್ಯ * ದತ್ತಾಂಶ, size_t ಸ್ಥಾನಾಂತರ, size_t size) {
    VK_LOG_DEBUG("ggml_backend_vk_buffer_get_tensor(" << ಬಫರ್ << ", " << tensor << ", " << ದತ್ತಾಂಶ << ", " << ಸ್ಥಾನಾಂತರ << ", " << size << ")");
    ggml_backend_vk_ಬಫರ್_context * buf_ctx = (ggml_backend_vk_ಬಫರ್_context *)ಬಫರ್->context;

    ಆದರೆ (size == 0) {
        ಹಿಂತಿರುಗಿ;
    }

    vk_ಬಫರ್ buf = buf_ctx->dev_ಬಫರ್;

    ggml_vk_ಬಫರ್_read(buf, vk_tensor_ಸ್ಥಾನಾಂತರ(tensor) + tensor->view_offs + ಸ್ಥಾನಾಂತರ, ದತ್ತಾಂಶ, size);
}

ಶೂನ್ಯ ggml_backend_vk_ಬಫರ್_get_tensor_2d(ggml_backend_ಬಫರ್_t ಬಫರ್, ಸ್ಥಿರಾಂಕ ggml_tensor * tensor, ಶೂನ್ಯ * ದತ್ತಾಂಶ, size_t ಸ್ಥಾನಾಂತರ,
                                                 size_t size, size_t n_copies, size_t stride_tensor, size_t stride_ದತ್ತಾಂಶ) {
    VK_LOG_DEBUG("ggml_backend_vk_buffer_get_tensor_2d(" << ಬಫರ್ << ", " << tensor << ", " << ದತ್ತಾಂಶ << ", " << ಸ್ಥಾನಾಂತರ << ", " << size << ", " <<
                 n_copies << ", " << stride_tensor << ", " << stride_ದತ್ತಾಂಶ << ")");
    ggml_backend_vk_ಬಫರ್_context * buf_ctx = (ggml_backend_vk_ಬಫರ್_context *)ಬಫರ್->context;

    ಆದರೆ (size == 0) {
        ಹಿಂತಿರುಗಿ;
    }

    vk_ಬಫರ್ buf = buf_ctx->dev_ಬಫರ್;

    ggml_vk_ಬಫರ್_read_2d(buf, vk_tensor_ಸ್ಥಾನಾಂತರ(tensor) + tensor->view_offs + ಸ್ಥಾನಾಂತರ, ದತ್ತಾಂಶ, stride_tensor, stride_ದತ್ತಾಂಶ, size, n_copies);
}

ತಾರ್ಕಿಕ ggml_backend_vk_ಬಫರ್_cpy_tensor(ggml_backend_ಬಫರ್_t ಬಫರ್, ಸ್ಥಿರಾಂಕ ggml_tensor * src, ggml_tensor * dst) {
    ಆದರೆ (ggml_nbytes(src) == 0) {
        ಹಿಂತಿರುಗಿ ನಿಜ;
    }

    ಆದರೆ (ggml_backend_ಬಫರ್_is_vk(src->ಬಫರ್)) {
        ggml_backend_vk_ಬಫರ್_context * src_buf_ctx = (ggml_backend_vk_ಬಫರ್_context *)src->ಬಫರ್->context;
        ggml_backend_vk_ಬಫರ್_context * dst_buf_ctx = (ggml_backend_vk_ಬಫರ್_context *)dst->ಬಫರ್->context;

        vk_ಬಫರ್ src_buf = src_buf_ctx->dev_ಬಫರ್;
        vk_ಬಫರ್ dst_buf = dst_buf_ctx->dev_ಬಫರ್;

        ggml_vk_ಬಫರ್_copy(dst_buf, vk_tensor_ಸ್ಥಾನಾಂತರ(dst) + dst->view_offs, src_buf, vk_tensor_ಸ್ಥಾನಾಂತರ(src) + src->view_offs, ggml_nbytes(src));

        ಹಿಂತಿರುಗಿ ನಿಜ;
    }
    ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;

    ಬಳಸದ(ಬಫರ್);
}

ಶೂನ್ಯ ggml_backend_vk_ಬಫರ್_clear(ggml_backend_ಬಫರ್_t ಬಫರ್, uint8_t value) {
    ggml_backend_vk_ಬಫರ್_context * ctx = (ggml_backend_vk_ಬಫರ್_context *)ಬಫರ್->context;

    ggml_vk_ಬಫರ್_memset(ctx->dev_ಬಫರ್, 0, value, ಬಫರ್->size);
}

ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * ggml_backend_vk_ಬಫರ್_type_name(ggml_backend_ಬಫರ್_type_t buft) {
    ggml_backend_vk_ಬಫರ್_type_context * ctx = (ggml_backend_vk_ಬಫರ್_type_context *)buft->context;

    ಹಿಂತಿರುಗಿ ctx->name.c_str();
}

ggml_backend_ಬಫರ್_t ggml_backend_vk_ಬಫರ್_type_alloc_ಬಫರ್(ggml_backend_ಬಫರ್_type_t buft, size_t size) {
    VK_LOG_ಸ್ಮರಣೆ("ggml_backend_vk_buffer_type_alloc_buffer(" << size << ")");
    ggml_backend_vk_ಬಫರ್_type_context * ctx = (ggml_backend_vk_ಬಫರ್_type_context *) buft->context;

    vk_ಬಫರ್ dev_ಬಫರ್ = ಶೂನ್ಯಸೂಚಕ;
    ಪ್ರಯತ್ನಿಸು {
        dev_ಬಫರ್ = ggml_vk_create_ಬಫರ್_device(ctx->device, size);
    } ಹಿಡಿ (ಸ್ಥಿರಾಂಕ vk::SystemError& e) {
        ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
    }

    ggml_backend_vk_ಬಫರ್_context * bufctx = ಹೊಸ ggml_backend_vk_ಬಫರ್_context(ctx->device, std::move(dev_ಬಫರ್), ctx->name);

    ಹಿಂತಿರುಗಿ ggml_backend_ಬಫರ್_init(buft, ggml_backend_vk_ಬಫರ್_interface, bufctx, size);
}

size_t ggml_backend_vk_ಬಫರ್_type_get_alignment(ggml_backend_ಬಫರ್_type_t buft) {
    ggml_backend_vk_ಬಫರ್_type_context * ctx = (ggml_backend_vk_ಬಫರ್_type_context *) buft->context;
    ಹಿಂತಿರುಗಿ ctx->device->properties.limits.min_Storage_ಬಫರ್_ಸ್ಥಾನಾಂತರ_Alignment;
}

size_t ggml_backend_vk_ಬಫರ್_type_get_max_ಗಾತ್ರ(ggml_backend_ಬಫರ್_type_t buft) {
    ggml_backend_vk_ಬಫರ್_type_context * ctx = (ggml_backend_vk_ಬಫರ್_type_context *) buft->context;
    ಹಿಂತಿರುಗಿ ctx->device->suballocation_ಖಂಡ_ಗಾತ್ರ;
}

size_t ggml_backend_vk_ಬಫರ್_type_get_alloc_ಗಾತ್ರ(ggml_backend_ಬಫರ್_type_t buft, ಸ್ಥಿರಾಂಕ ggml_tensor * tensor) {
    ಹಿಂತಿರುಗಿ ggml_nbytes(tensor);

    ಬಳಸದ(buft);
}

ggml_backend_ಬಫರ್_type_t ggml_backend_vk_ಬಫರ್_type(size_t dev_ಅಂಕೆ) {
    ggml_vk_instance_init();

    VK_LOG_DEBUG("ggml_backend_vk_buffer_type(" << dev_ಅಂಕೆ << ")");

    vk_device dev = ggml_vk_get_device(dev_ಅಂಕೆ);

    ಹಿಂತಿರುಗಿ &dev->ಬಫರ್_type;
}

ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * ggml_backend_vk_host_ಬಫರ್_type_name(ggml_backend_ಬಫರ್_type_t buft) {
    ಹಿಂತಿರುಗಿ GGML_VK_NAME "_Host";

    ಬಳಸದ(buft);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_vk_host_ಬಫರ್_free_ಬಫರ್(ggml_backend_ಬಫರ್_t ಬಫರ್) {
    VK_LOG_ಸ್ಮರಣೆ("ggml_backend_vk_host_buffer_free_buffer()");
    ggml_vk_host_free(ಬಫರ್->context);
}

ಸ್ಥಿರ ggml_backend_ಬಫರ್_t ggml_backend_vk_host_ಬಫರ್_type_alloc_ಬಫರ್(ggml_backend_ಬಫರ್_type_t buft, size_t size) {
    VK_LOG_ಸ್ಮರಣೆ("ggml_backend_vk_host_buffer_type_alloc_buffer(" << size << ")");

    size += 32;  // Behave like the CPU buffer type
    ಶೂನ್ಯ * ptr = ggml_vk_host_malloc(size);
    ಆದರೆ (ptr == ಶೂನ್ಯಸೂಚಕ) {
        GGML_LOG_WARN("ggml_vulkan: Failed to allocate pinned memory\n");
        // fallback to cpu buffer
        ಹಿಂತಿರುಗಿ ggml_backend_buft_alloc_ಬಫರ್(ggml_backend_cpu_ಬಫರ್_type(), size);
    }

    ggml_backend_ಬಫರ್_t ಬಫರ್ = ggml_backend_cpu_ಬಫರ್_from_ptr(ptr, size);
    ಬಫರ್->buft = buft;
    ಬಫರ್->iface.free_ಬಫರ್ = ggml_backend_vk_host_ಬಫರ್_free_ಬಫರ್;

    ಹಿಂತಿರುಗಿ ಬಫರ್;

    ಬಳಸದ(buft);
}

ಸ್ಥಿರ size_t ggml_backend_vk_host_ಬಫರ್_type_get_alignment(ggml_backend_ಬಫರ್_type_t buft) {
    ಹಿಂತಿರುಗಿ vk_instance.devices[0]->properties.limits.min_ಸ್ಮರಣೆ_Map_Alignment;

    ಬಳಸದ(buft);
}

ಸ್ಥಿರ size_t ggml_backend_vk_host_ಬಫರ್_type_get_max_ಗಾತ್ರ(ggml_backend_ಬಫರ್_type_t buft) {
    ಹಿಂತಿರುಗಿ vk_instance.devices[0]->suballocation_ಖಂಡ_ಗಾತ್ರ;

    ಬಳಸದ(buft);
}

ggml_backend_ಬಫರ್_type_t ggml_backend_vk_host_ಬಫರ್_type() {
    ಸ್ಥಿರ ರಚನೆ ggml_backend_ಬಫರ್_type ggml_backend_vk_ಬಫರ್_type_host = {
        /* .iface    = */ {
            /* .get_name            = */ ggml_backend_vk_host_ಬಫರ್_type_name,
            /* .alloc_buffer        = */ ggml_backend_vk_host_ಬಫರ್_type_alloc_ಬಫರ್,
            /* .alloc_buffer_n      = */ ಶೂನ್ಯಸೂಚಕ,
            /* .get_alignment       = */ ggml_backend_vk_host_ಬಫರ್_type_get_alignment,
            /* .get_max_size        = */ ggml_backend_vk_host_ಬಫರ್_type_get_max_ಗಾತ್ರ,
            /* .get_alloc_size      = */ ggml_backend_cpu_ಬಫರ್_type()->iface.get_alloc_ಗಾತ್ರ,
            /* .get_alloc_size_n    = */ ಖಾಲಿ,
            /* .is_host             = */ ggml_backend_cpu_ಬಫರ್_type()->iface.is_host,
        },
        /* .device   = */ ggml_backend_reg_dev_get(ggml_backend_vk_reg(), 0),
        /* .context  = */ ಶೂನ್ಯಸೂಚಕ,
    };

    // Make sure device 0 is initialized
    ggml_vk_instance_init();
    ggml_vk_get_device(0);

    ಹಿಂತಿರುಗಿ &ggml_backend_vk_ಬಫರ್_type_host;
}

ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * ggml_backend_vk_name(ggml_backend_t backend) {
    ggml_backend_vk_context * ctx = (ggml_backend_vk_context *)backend->context;

    ಹಿಂತಿರುಗಿ ctx->name.c_str();
}

ಶೂನ್ಯ ggml_backend_vk_free(ggml_backend_t backend) {
    ggml_backend_vk_context * ctx = (ggml_backend_vk_context *)backend->context;
    VK_LOG_DEBUG("ggml_backend_vk_free(" << ctx->name << ")");

    ggml_vk_cleanup(ctx);

    ಅಳಿಸು ctx;
    ಅಳಿಸು backend;
}

ಸ್ಥಿರ ggml_backend_ಬಫರ್_type_t ggml_backend_vk_get_default_ಬಫರ್_type(ggml_backend_t backend) {
    ggml_backend_vk_context * ctx = (ggml_backend_vk_context *)backend->context;

    ಹಿಂತಿರುಗಿ &ctx->device->ಬಫರ್_type;
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_vk_ಗಣ_tensor_2d_async(ggml_backend_t backend, ggml_tensor * tensor, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ದತ್ತಾಂಶ, size_t ಸ್ಥಾನಾಂತರ,
                                                size_t size, size_t n_copies, size_t stride_tensor, size_t stride_ದತ್ತಾಂಶ) {
    VK_LOG_DEBUG("ggml_backend_vk_set_tensor_2d_async(" << size << ", " << n_copies << ")");
    ggml_backend_vk_context * ctx = (ggml_backend_vk_context *)backend->context;
    GGML_ASSERT((tensor->ಬಫರ್->buft == ggml_backend_vk_get_default_ಬಫರ್_type(backend) || tensor->ಬಫರ್->buft == ggml_backend_vk_host_ಬಫರ್_type()) && "unsupported buffer type");

    ಆದರೆ (size == 0) {
        ಹಿಂತಿರುಗಿ;
    }

    ggml_backend_vk_ಬಫರ್_context * buf_ctx = (ggml_backend_vk_ಬಫರ್_context *)tensor->ಬಫರ್->context;

    vk_context cpy_ctx;

    ಆದರೆ (ctx->device->async_use_transfer_ಸರತಿ) {
        cpy_ctx = ggml_vk_get_transfer_ctx(ctx);
    } ಇಲ್ಲದಿದ್ದರೆ {
        cpy_ctx = ggml_vk_get_compute_ctx(ctx);
    }

    vk_ಬಫರ್ buf = buf_ctx->dev_ಬಫರ್;

    ಸ್ವಯಂಚಾಲಿತ dst_ಸ್ಥಾನಾಂತರ = vk_tensor_ಸ್ಥಾನಾಂತರ(tensor) + tensor->view_offs + ಸ್ಥಾನಾಂತರ;

    ತಾರ್ಕಿಕ ret = ggml_vk_ಬಫರ್_write_2d_async(cpy_ctx, buf, dst_ಸ್ಥಾನಾಂತರ, ದತ್ತಾಂಶ, stride_ದತ್ತಾಂಶ, stride_tensor, size, n_copies);

    ಆದರೆ (!ret) {
        ಸ್ಥಿರಾಂಕ size_t staging_ಗಾತ್ರ = size * n_copies;
        ggml_vk_ensure_sync_staging_ಬಫರ್(ctx, staging_ಗಾತ್ರ);
        ggml_vk_sync_buffers(ಶೂನ್ಯಸೂಚಕ, cpy_ctx);

        std::vector<vk::ಬಫರ್_Copy> slices(1);
        ಆದರೆ (size == stride_tensor) {
            slices[0].src_ಸ್ಥಾನಾಂತರ_2 = 0;
            slices[0].dst_ಸ್ಥಾನಾಂತರ_2 = dst_ಸ್ಥಾನಾಂತರ;
            slices[0].size = staging_ಗಾತ್ರ;
        } ಇಲ್ಲದಿದ್ದರೆ {
            slices.resize(n_copies);
            ಗಾಗಿ (size_t i = 0; i < n_copies; i++) {
                slices[i].src_ಸ್ಥಾನಾಂತರ_2 = i * size;
                slices[i].dst_ಸ್ಥಾನಾಂತರ_2 = dst_ಸ್ಥಾನಾಂತರ + i * stride_tensor;
                slices[i].size = size;
            }
        }

        cpy_ctx->s->ಬಫರ್->buf.copy_ಬಫರ್(ctx->sync_staging->ಬಫರ್, buf->ಬಫರ್, slices);

        ಆದರೆ (size == stride_ದತ್ತಾಂಶ) {
            deferred_memcpy(ctx->sync_staging->ptr, ದತ್ತಾಂಶ, staging_ಗಾತ್ರ, &cpy_ctx->ಒಳ_memcpys);
        } ಇಲ್ಲದಿದ್ದರೆ {
            ಗಾಗಿ (size_t i = 0; i < n_copies; i++) {
                deferred_memcpy((uint8_t *)ctx->sync_staging->ptr + i * size, (ಸ್ಥಿರಾಂಕ uint8_t *)ದತ್ತಾಂಶ + i * stride_ದತ್ತಾಂಶ, size, &cpy_ctx->ಒಳ_memcpys);
            }
        }
        ggml_vk_synchronize(ctx);
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_vk_ಗಣ_tensor_async(ggml_backend_t backend, ggml_tensor * tensor, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ದತ್ತಾಂಶ, size_t ಸ್ಥಾನಾಂತರ, size_t size) {
    VK_LOG_DEBUG("ggml_backend_vk_set_tensor_async(" << size << ")");
    ggml_backend_vk_ಗಣ_tensor_2d_async(backend, tensor, ದತ್ತಾಂಶ, ಸ್ಥಾನಾಂತರ, size, 1, size, size);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_vk_get_tensor_2d_async(ggml_backend_t backend, ಸ್ಥಿರಾಂಕ ggml_tensor * tensor, ಶೂನ್ಯ * ದತ್ತಾಂಶ, size_t ಸ್ಥಾನಾಂತರ,
                                                size_t size, size_t n_copies, size_t stride_tensor, size_t stride_ದತ್ತಾಂಶ) {
    VK_LOG_DEBUG("ggml_backend_vk_get_tensor_2d_async(" << size << ", " << n_copies << ")");
    ggml_backend_vk_context * ctx = (ggml_backend_vk_context *)backend->context;
    GGML_ASSERT((tensor->ಬಫರ್->buft == ggml_backend_vk_get_default_ಬಫರ್_type(backend) || tensor->ಬಫರ್->buft == ggml_backend_vk_host_ಬಫರ್_type()) && "unsupported buffer type");

    ಆದರೆ (size == 0) {
        ಹಿಂತಿರುಗಿ;
    }

    ggml_backend_vk_ಬಫರ್_context * buf_ctx = (ggml_backend_vk_ಬಫರ್_context *)tensor->ಬಫರ್->context;

    vk_context compute_ctx = ggml_vk_get_compute_ctx(ctx);

    vk_ಬಫರ್ buf = buf_ctx->dev_ಬಫರ್;

    ಸ್ವಯಂಚಾಲಿತ src_ಸ್ಥಾನಾಂತರ = vk_tensor_ಸ್ಥಾನಾಂತರ(tensor) + tensor->view_offs + ಸ್ಥಾನಾಂತರ;
    ತಾರ್ಕಿಕ ret = ggml_vk_ಬಫರ್_read_2d_async(compute_ctx, buf, src_ಸ್ಥಾನಾಂತರ, ದತ್ತಾಂಶ, stride_tensor, stride_ದತ್ತಾಂಶ, size, n_copies);

    ಆದರೆ (!ret) {
        ಸ್ಥಿರಾಂಕ size_t staging_ಗಾತ್ರ = size * n_copies;
        ggml_vk_ensure_sync_staging_ಬಫರ್(ctx, staging_ಗಾತ್ರ);
        ggml_vk_sync_buffers(ಶೂನ್ಯಸೂಚಕ, compute_ctx);

        std::vector<vk::ಬಫರ್_Copy> slices(1);
        ಆದರೆ (size == stride_tensor) {
            slices[0].src_ಸ್ಥಾನಾಂತರ_2 = src_ಸ್ಥಾನಾಂತರ;
            slices[0].dst_ಸ್ಥಾನಾಂತರ_2 = 0;
            slices[0].size = staging_ಗಾತ್ರ;
        } ಇಲ್ಲದಿದ್ದರೆ {
            slices.resize(n_copies);
            ಗಾಗಿ (size_t i = 0; i < n_copies; i++) {
                slices[i].src_ಸ್ಥಾನಾಂತರ_2 = src_ಸ್ಥಾನಾಂತರ + i * stride_tensor;
                slices[i].dst_ಸ್ಥಾನಾಂತರ_2 = i * size;
                slices[i].size = size;
            }
        }

        compute_ctx->s->ಬಫರ್->buf.copy_ಬಫರ್(buf->ಬಫರ್, ctx->sync_staging->ಬಫರ್, slices);

        ಆದರೆ (size == stride_ದತ್ತಾಂಶ) {
            deferred_memcpy(ದತ್ತಾಂಶ, ctx->sync_staging->ptr, staging_ಗಾತ್ರ, &compute_ctx->ಹೊರಹೋಗು_memcpys);
        } ಇಲ್ಲದಿದ್ದರೆ {
            ಗಾಗಿ (size_t i = 0; i < n_copies; i++) {
                deferred_memcpy((uint8_t *)ದತ್ತಾಂಶ + i * stride_ದತ್ತಾಂಶ, (ಸ್ಥಿರಾಂಕ uint8_t *)ctx->sync_staging->ptr + i * size, size, &compute_ctx->ಹೊರಹೋಗು_memcpys);
            }
        }
        ggml_vk_synchronize(ctx);
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_vk_get_tensor_async(ggml_backend_t backend, ಸ್ಥಿರಾಂಕ ggml_tensor * tensor, ಶೂನ್ಯ * ದತ್ತಾಂಶ, size_t ಸ್ಥಾನಾಂತರ, size_t size) {
    VK_LOG_DEBUG("ggml_backend_vk_get_tensor_async(" << size << ")");
    ggml_backend_vk_get_tensor_2d_async(backend, tensor, ದತ್ತಾಂಶ, ಸ್ಥಾನಾಂತರ, size, 1, size, size);
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_backend_vk_cpy_tensor_async(ggml_backend_t backend_src, ggml_backend_t backend_dst, ಸ್ಥಿರಾಂಕ ggml_tensor * src, ggml_tensor * dst) {
    VK_LOG_DEBUG("ggml_backend_vk_cpy_tensor_async(" << src << " -> " << dst << ", size=" << ggml_nbytes(src) << ")");
    ggml_backend_vk_context * ctx = (ggml_backend_vk_context *)backend_dst->context;

    // Skip zero-size tensors
    ಆದರೆ (ggml_nbytes(src) == 0) {
        ಹಿಂತಿರುಗಿ ನಿಜ;
    }

    ಆದರೆ (dst->ಬಫರ್->buft != ggml_backend_vk_get_default_ಬಫರ್_type(backend_dst)) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ggml_backend_vk_ಬಫರ್_context * dst_buf_ctx = (ggml_backend_vk_ಬಫರ್_context *)dst->ಬಫರ್->context;
    vk_ಬಫರ್ dst_buf = dst_buf_ctx->dev_ಬಫರ್;

    ಆದರೆ (ggml_backend_ಬಫರ್_is_vk(src->ಬಫರ್)) {
        ggml_backend_vk_ಬಫರ್_context * src_buf_ctx = (ggml_backend_vk_ಬಫರ್_context *)src->ಬಫರ್->context;

        // Async copy only works within the same device
        ಆದರೆ (src_buf_ctx->dev_ಬಫರ್->device != dst_buf->device) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }

        vk_context compute_ctx = ggml_vk_get_compute_ctx(ctx);

        ggml_vk_ಬಫರ್_copy_async(compute_ctx, dst_buf, vk_tensor_ಸ್ಥಾನಾಂತರ(dst) + dst->view_offs,
                                   src_buf_ctx->dev_ಬಫರ್, vk_tensor_ಸ್ಥಾನಾಂತರ(src) + src->view_offs,
                                   ggml_nbytes(src));
        ಹಿಂತಿರುಗಿ ನಿಜ;
    }

    ಆದರೆ (ggml_backend_ಬಫರ್_is_host(src->ಬಫರ್)) {
        vk_ಬಫರ್ pinned_buf = ಶೂನ್ಯಸೂಚಕ;
        size_t pinned_ಸ್ಥಾನಾಂತರ = 0;
        ggml_vk_host_get(ctx->device, src->ದತ್ತಾಂಶ, pinned_buf, pinned_ಸ್ಥಾನಾಂತರ);
        ಆದರೆ (pinned_buf == ಶೂನ್ಯಸೂಚಕ) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }

        // If the backend is idle, use a CPU copy to avoid GPU synchronization overhead.
        ಸ್ಥಿರ constexpr size_t max_cpu_copy_ಗಾತ್ರ = 128 * 1024;
        ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ src_backend_synchronous = backend_src->iface.synchronize == ಶೂನ್ಯಸೂಚಕ;
        ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ transfer_idle = !ctx->device->async_use_transfer_ಸರತಿ ||
                                   ctx->transfer_semaphore_ಕೊನೆಯ_submitted == ctx->transfer_semaphore.value;
        ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ backend_idle = ctx->compute_ctx.expired() && ctx->transfer_ctx.expired() &&
                                  !ctx->submit_pending && !ctx->almost_ready_fence_pending && transfer_idle;
        ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ dst_host_coherent =
            (dst_buf->ಸ್ಮರಣೆ_property_flags & (vk::ಸ್ಮರಣೆ_Property_Flag_ಬಿಟ್ಟುಗಳು::eHostVisible | vk::ಸ್ಮರಣೆ_Property_Flag_ಬಿಟ್ಟುಗಳು::eHostCoherent)) ==
            (vk::ಸ್ಮರಣೆ_Property_Flag_ಬಿಟ್ಟುಗಳು::eHostVisible | vk::ಸ್ಮರಣೆ_Property_Flag_ಬಿಟ್ಟುಗಳು::eHostCoherent);

        ಆದರೆ ((backend_src == backend_dst || src_backend_synchronous) && backend_idle && dst_host_coherent && ggml_nbytes(src) <= max_cpu_copy_ಗಾತ್ರ) {
            ggml_vk_ಬಫರ್_write(dst_buf, vk_tensor_ಸ್ಥಾನಾಂತರ(dst) + dst->view_offs, src->ದತ್ತಾಂಶ, ggml_nbytes(src));
            ಹಿಂತಿರುಗಿ ನಿಜ;
        }

        vk_context cpy_ctx;
        ಆದರೆ (ctx->device->async_use_transfer_ಸರತಿ) {
            cpy_ctx = ggml_vk_get_transfer_ctx(ctx);
        } ಇಲ್ಲದಿದ್ದರೆ {
            cpy_ctx = ggml_vk_get_compute_ctx(ctx);
        }

        ಹಿಂತಿರುಗಿ ggml_vk_ಬಫರ್_write_async(cpy_ctx, dst_buf,
                                          vk_tensor_ಸ್ಥಾನಾಂತರ(dst) + dst->view_offs,
                                          src->ದತ್ತಾಂಶ, ggml_nbytes(src));
    }

    ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
}

ಶೂನ್ಯ ggml_vk_synchronize(ggml_backend_vk_context * ctx) {
    VK_LOG_DEBUG("ggml_vk_synchronize()");

    ತಾರ್ಕಿಕ do_transfer = !ctx->compute_ctx.expired();

    ಆದರೆ (ggml_vk_submit_transfer_ctx(ctx)) {
        ctx->submit_pending = ನಿಜ;
    }

    vk_context compute_ctx;
    vk_command_ಬಫರ್* cmd_buf = ಶೂನ್ಯಸೂಚಕ;
    ಆದರೆ (do_transfer) {
        compute_ctx = ctx->compute_ctx.ಬೀಗ();
        ಆದರೆ (compute_ctx->s) {
            cmd_buf = compute_ctx->s->ಬಫರ್;
        }

        ggml_vk_ctx_ಕೊನೆ(compute_ctx);

        ಗಾಗಿ (ಸ್ವಯಂಚಾಲಿತ& cpy : compute_ctx->ಒಳ_memcpys) {
            ಸ್ಮರಣೆ_ನಕಲು(cpy.dst, cpy.src, cpy.n);
        }

        ಆದರೆ (ctx->device->serialize_submissions) {
            ggml_vk_submit(compute_ctx, ctx->fence);
            VK_CHECK(ctx->device->device.waitForFences({ ctx->fence }, ನಿಜ, UINT64_MAX), "synchronize waitForFences", ctx->device);
            ctx->device->device.resetFences({ ctx->fence });
        } ಇಲ್ಲದಿದ್ದರೆ {
            ggml_vk_submit(compute_ctx, {});
        }
        ctx->submit_pending = ನಿಜ;
    }

    ಆದರೆ (ctx->submit_pending) {
        ಆದರೆ (ctx->device->serialize_submissions) {
            ctx->submit_pending = ಸುಳ್ಳು;
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (ctx->device->async_use_transfer_ಸರತಿ && ctx->transfer_semaphore_ಕೊನೆಯ_submitted < ctx->transfer_semaphore.value) {
            vk::TimelineSemaphoreSubmitInfo tl_info{
                1, &ctx->transfer_semaphore.value,
                0, ಶೂನ್ಯಸೂಚಕ,
            };
            vk::PipelineStageFlags stage = ctx->device->transfer_ಸರತಿ->stage_flags;
            vk::SubmitInfo si{
                1, &ctx->transfer_semaphore.s, &stage,
                0, ಶೂನ್ಯಸೂಚಕ,
                0, ಶೂನ್ಯಸೂಚಕ,
            };
            si.ಗಣ_PNext(&tl_info);
            ctx->device->compute_ಸರತಿ->handle->submit({ si }, ctx->fence);
            ctx->transfer_semaphore_ಕೊನೆಯ_submitted = ctx->transfer_semaphore.value;
        } ಇಲ್ಲದಿದ್ದರೆ {
            ctx->device->compute_ಸರತಿ->handle->submit({}, ctx->fence);
        }
        ಆದರೆ (!ctx->device->serialize_submissions) {
            ggml_vk_wait_for_fence(ctx);
        }
        ctx->submit_pending = ಸುಳ್ಳು;
        ಆದರೆ (cmd_buf) {
            cmd_buf->ಒಳ_use = ಸುಳ್ಳು;
            cmd_buf->buf.reset();
        }
    }

    ಆದರೆ (do_transfer) {
        ಗಾಗಿ (ಸ್ವಯಂಚಾಲಿತ& cpy : compute_ctx->ಹೊರಹೋಗು_memcpys) {
            ಸ್ಮರಣೆ_ನಕಲು(cpy.dst, cpy.src, cpy.n);
        }
        ctx->compute_ctx.reset();
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_vk_synchronize(ggml_backend_t backend) {
    VK_LOG_DEBUG("ggml_backend_vk_synchronize()");
    ggml_backend_vk_context * ctx = (ggml_backend_vk_context *)backend->context;

    ggml_vk_synchronize(ctx);

    ggml_vk_ನಕ್ಷೆ_cleanup(ctx);
}

ತಾರ್ಕಿಕ ggml_vk_is_ಖಾಲಿಗಳು(ggml_tensor * ಗಂಟು) {
    ಹಿಂತಿರುಗಿ ggml_is_ಖಾಲಿಗಳು(ಗಂಟು) || ಗಂಟು->op == GGML_OP_NONE || ಗಂಟು->op == GGML_OP_RESHAPE || ಗಂಟು->op == GGML_OP_TRANSPOSE || ಗಂಟು->op == GGML_OP_VIEW || ಗಂಟು->op == GGML_OP_ಕ್ರಮಬದಲಾವಣೆ;
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_vk_can_fuse_unary_mul(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_cgraph * cgraph, ಪೂರ್ಣಾಂಕ unary_idx, ಪೂರ್ಣಾಂಕ mul_idx) {
    ಸ್ಥಿರಾಂಕ ggml_tensor * unary = cgraph->nodes[unary_idx];
    ಸ್ಥಿರಾಂಕ ggml_tensor * mul = cgraph->nodes[mul_idx];

    ಆದರೆ (ggml_vk_unary_mul_op_ಸೂಚ್ಯಂಕ(ggml_get_unary_op(unary)) < 0) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
    ಆದರೆ (unary->type != GGML_TYPE_F32 && unary->type != GGML_TYPE_F16) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
    ಆದರೆ (unary->type != mul->type) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
    ಆದರೆ (mul->src[0] != unary && mul->src[1] != unary) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
    ಸ್ಥಿರಾಂಕ ggml_tensor * other = (mul->src[0] == unary) ? mul->src[1] : mul->src[0];
    ಆದರೆ (other == ಶೂನ್ಯಸೂಚಕ || other->type != unary->type) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
    ಆದರೆ (!ggml_is_contiguous_1(other) || !ggml_is_contiguous_1(unary->src[0])) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
    // fastmod needs src to tile into dst
    ಆದರೆ (mul->src[0] == unary) {
        ಹಿಂತಿರುಗಿ ggml_can_repeat(other, unary);
    }
    ಹಿಂತಿರುಗಿ ggml_can_repeat(unary, mul->src[0]);
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_vk_can_fuse_unary_mul_pair(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_cgraph * cgraph, ಪೂರ್ಣಾಂಕ ಗಂಟು_idx) {
    ಸ್ಥಿರಾಂಕ ಎಣಿಕೆ ggml_op ops[]    = { GGML_OP_UNARY, GGML_OP_MUL };
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ           outputs[] = { ಗಂಟು_idx + 1 };
    ಹಿಂತಿರುಗಿ ggml_can_fuse_subgraph(cgraph, ಗಂಟು_idx, 2, ops, outputs, 1) &&
           ggml_vk_can_fuse_unary_mul(cgraph, ಗಂಟು_idx, ಗಂಟು_idx + 1);
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_vk_can_fuse_hc_post_gate(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_cgraph * cgraph, ಪೂರ್ಣಾಂಕ ಗಂಟು_idx) {
    ಸ್ಥಿರಾಂಕ ggml_tensor * scale_ಒಳ  = cgraph->nodes[ಗಂಟು_idx];
    ಸ್ಥಿರಾಂಕ ggml_tensor * sigmoid   = cgraph->nodes[ಗಂಟು_idx + 1];
    ಸ್ಥಿರಾಂಕ ggml_tensor * scale_ಹೊರಹೋಗು = cgraph->nodes[ಗಂಟು_idx + 2];

    // the shader folds scale -> sigmoid -> scale; a bias on either scale is not handled
    ಹಿಂತಿರುಗಿ ggml_get_unary_op(sigmoid) == GGML_UNARY_OP_SIGMOID &&
           ggml_get_op_params_f32(scale_ಒಳ, 1) == 0.0f &&
           ggml_get_op_params_f32(scale_ಹೊರಹೋಗು, 1) == 0.0f &&
           scale_ಒಳ->src[0]->type == GGML_TYPE_F32 &&
           ggml_are_same_shape(scale_ಒಳ->src[0], scale_ಹೊರಹೋಗು);
}

ತಾರ್ಕಿಕ ggml_vk_can_fuse(ಸ್ಥಿರಾಂಕ ggml_backend_vk_context * ctx, ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_cgraph * cgraph, ಪೂರ್ಣಾಂಕ ಗಂಟು_idx, std::initializer_list<ಎಣಿಕೆ ggml_op> ops) {
    ಆದರೆ (ops.size() == 2 && ops.begin()[0] == GGML_OP_UNARY && ops.begin()[1] == GGML_OP_MUL) {
        ಹಿಂತಿರುಗಿ ggml_vk_can_fuse_unary_mul_pair(cgraph, ಗಂಟು_idx);
    }

    ಆದರೆ (!ggml_can_fuse(cgraph, ಗಂಟು_idx, ops)) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಆದರೆ ((ops.size() == 2 || ops.size() == 3 || ops.size() == 4) &&
        ops.begin()[0] == GGML_OP_RMS_NORM && ops.begin()[1] == GGML_OP_MUL) {
        // additional constraints specific to this fusion
        ಸ್ಥಿರಾಂಕ ggml_tensor *rms_norm = cgraph->nodes[ಗಂಟು_idx];
        ಸ್ಥಿರಾಂಕ ggml_tensor *mul = cgraph->nodes[ಗಂಟು_idx + 1];

        GGML_ASSERT(rms_norm->src[0]->type == GGML_TYPE_F32);
        GGML_ASSERT(rms_norm->type == GGML_TYPE_F32);
        // rms_norm only supports f32
        ಆದರೆ (mul->src[0]->type != GGML_TYPE_F32 ||
            mul->src[1]->type != GGML_TYPE_F32 ||
            mul->type != GGML_TYPE_F32) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }
        // if rms_norm is the B operand, then we don't handle broadcast
        ಆದರೆ (rms_norm == mul->src[1] &&
            !ggml_are_same_shape(mul->src[0], rms_norm)) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }
        // rms_norm shader assumes contiguous rows
        ಆದರೆ (!ggml_is_contiguous_ಅಡ್ಡಸಾಲುಗಳು(mul->src[0]) || !ggml_is_contiguous_ಅಡ್ಡಸಾಲುಗಳು(mul->src[1])) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }

        ಆದರೆ (ops.size() >= 3 && ops.begin()[2] == GGML_OP_ಕೂಡಿಸು) {
            ಸ್ಥಿರಾಂಕ ggml_tensor *ಕೂಡಿಸು_2 = cgraph->nodes[ಗಂಟು_idx + 2];
            ಸ್ಥಿರಾಂಕ ggml_tensor *residual = ಕೂಡಿಸು_2->src[0] == mul ? ಕೂಡಿಸು_2->src[1] : ಕೂಡಿಸು_2->src[0];
            ಆದರೆ (ಕೂಡಿಸು_2->src[0] != mul && ಕೂಡಿಸು_2->src[1] != mul) {
                ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
            }
            ಆದರೆ (residual->type != GGML_TYPE_F32 || ಕೂಡಿಸು_2->type != GGML_TYPE_F32 ||
                !ggml_are_same_shape(ಕೂಡಿಸು_2, residual) || !ggml_is_contiguous(residual) ||
                !ggml_is_contiguous(ಕೂಡಿಸು_2) || get_misalign_ಬೈಟುಗಳು(ctx, residual) != 0) {
                ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
            }

            ಸ್ಥಿರಾಂಕ ggml_tensor *dst = ಕೂಡಿಸು_2;
            ಆದರೆ (ops.size() == 4) {
                ಆದರೆ (ops.begin()[3] != GGML_OP_MUL) {
                    ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
                }

                ಸ್ಥಿರಾಂಕ ggml_tensor *post_mul = cgraph->nodes[ಗಂಟು_idx + 3];
                ಸ್ಥಿರಾಂಕ ggml_tensor *scale = post_mul->src[0] == ಕೂಡಿಸು_2 ? post_mul->src[1] : post_mul->src[0];
                ಆದರೆ (post_mul->src[0] != ಕೂಡಿಸು_2 && post_mul->src[1] != ಕೂಡಿಸು_2) {
                    ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
                }
                // The shader reads data_e[0], so the final multiply must use a scalar.
                ಆದರೆ (scale->type != GGML_TYPE_F32 || post_mul->type != GGML_TYPE_F32 ||
                    ggml_nelements(scale) != 1 || !ggml_is_contiguous(post_mul) ||
                    get_misalign_ಬೈಟುಗಳು(ctx, scale) != 0) {
                    ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
                }
                dst = post_mul;
            }

            ಆದರೆ (get_misalign_ಬೈಟುಗಳು(ctx, dst) != 0) {
                ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
            }
        }
    }

    ಸ್ವಯಂಚಾಲಿತ ಸ್ಥಿರಾಂಕ &mm_ಕೂಡಿಸು_ok = [&](ಸ್ಥಿರಾಂಕ ggml_tensor *mul, ಸ್ಥಿರಾಂಕ ggml_tensor *ಕೂಡಿಸು_2) {
        ಸ್ಥಿರಾಂಕ ggml_tensor *bias = ಕೂಡಿಸು_2->src[0] == mul ? ಕೂಡಿಸು_2->src[1] : ಕೂಡಿಸು_2->src[0];

        // mat-vec only
        ಆದರೆ (ggml_nrows(mul) != 1) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }
        // shaders assume the types match
        ಆದರೆ (mul->type != bias->type) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }
        // shaders reuse the D shape for bias
        ಆದರೆ (!ggml_are_same_shape(mul, bias) ||
            !ggml_are_same_stride(mul, bias)) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }
        // unaligned bias isn't handled
        ಆದರೆ (get_misalign_ಬೈಟುಗಳು(ctx, bias) != 0) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }
        ಹಿಂತಿರುಗಿ ನಿಜ;
    };

    ಆದರೆ ((ops.size() == 2 || ops.size() == 3) && ops.begin()[0] == GGML_OP_MUL_MAT && ops.begin()[1] == GGML_OP_ಕೂಡಿಸು) {
        // additional constraints specific to this fusion
        ಸ್ಥಿರಾಂಕ ggml_tensor *mul = cgraph->nodes[ಗಂಟು_idx];
        ಸ್ಥಿರಾಂಕ ggml_tensor *ಕೂಡಿಸು_2 = cgraph->nodes[ಗಂಟು_idx + 1];

        ಆದರೆ (!mm_ಕೂಡಿಸು_ok(mul, ಕೂಡಿಸು_2)) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }
        ಆದರೆ (ops.size() == 3) {
            ಆದರೆ (ops.begin()[2] != GGML_OP_ಕೂಡಿಸು) {
                ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
            }
            ಆದರೆ (!mm_ಕೂಡಿಸು_ok(ಕೂಡಿಸು_2, cgraph->nodes[ಗಂಟು_idx + 2])) {
                ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
            }
        }
    }

    ಸ್ವಯಂಚಾಲಿತ ಸ್ಥಿರಾಂಕ &mmid_mul_ok = [&](ಸ್ಥಿರಾಂಕ ggml_tensor *mmid, ಸ್ಥಿರಾಂಕ ggml_tensor *mul) {
        ಸ್ಥಿರಾಂಕ ggml_tensor *scale = mul->src[1];

        ಆದರೆ (mmid != mul->src[0]) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }
        // mat-vec only
        ಆದರೆ (!ggml_vk_use_mul_mat_vec_id(cgraph, ಗಂಟು_idx)) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }
        // shaders assume the types match
        ಆದರೆ (mmid->type != scale->type) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }
        // shaders assume the bias is contiguous
        ಆದರೆ (!ggml_is_contiguous(scale)) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }
        // unaligned bias isn't handled
        ಆದರೆ (get_misalign_ಬೈಟುಗಳು(ctx, scale) != 0) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }
        // shader only indexes by expert index
        ಆದರೆ (scale->ne[0] != 1 ||
            scale->ne[1] != mul->ne[1] ||
            scale->ne[2] != 1 ||
            scale->ne[3] != 1) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }
        ಹಿಂತಿರುಗಿ ನಿಜ;
    };

    ಆದರೆ ((ops.size() == 2 || ops.size() == 3) && ops.begin()[0] == GGML_OP_MUL_MAT_ID && ops.begin()[1] == GGML_OP_ಕೂಡಿಸು_ID) {
        // additional constraints specific to this fusion
        ಸ್ಥಿರಾಂಕ ggml_tensor *mul = cgraph->nodes[ಗಂಟು_idx];
        ಸ್ಥಿರಾಂಕ ggml_tensor *ಕೂಡಿಸು_2 = cgraph->nodes[ಗಂಟು_idx + 1];
        ಸ್ಥಿರಾಂಕ ggml_tensor *bias = ಕೂಡಿಸು_2->src[1];

        ಆದರೆ (mul != ಕೂಡಿಸು_2->src[0]) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }
        // mat-vec only
        ಆದರೆ (!ggml_vk_use_mul_mat_vec_id(cgraph, ಗಂಟು_idx)) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }
        // shaders assume the types match
        ಆದರೆ (mul->type != bias->type) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }
        // shaders assume the bias is contiguous
        ಆದರೆ (!ggml_is_contiguous(bias)) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }
        // the ID tensor must be the same for mul_mat_id and add_id
        ಆದರೆ (mul->src[2] != ಕೂಡಿಸು_2->src[2]) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }
        // unaligned bias isn't handled
        ಆದರೆ (get_misalign_ಬೈಟುಗಳು(ctx, bias) != 0) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }

        ಆದರೆ (ops.size() == 3) {
            ಆದರೆ (ops.begin()[2] != GGML_OP_MUL) {
                ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
            }
            ಸ್ಥಿರಾಂಕ ggml_tensor *mul = cgraph->nodes[ಗಂಟು_idx + 2];
            ಹಿಂತಿರುಗಿ mmid_mul_ok(ಕೂಡಿಸು_2, mul);
        }
    }

    ಆದರೆ (ops.size() == 2 && ops.begin()[0] == GGML_OP_MUL_MAT_ID && ops.begin()[1] == GGML_OP_MUL) {
        // additional constraints specific to this fusion
        ಸ್ಥಿರಾಂಕ ggml_tensor *mmid = cgraph->nodes[ಗಂಟು_idx];
        ಸ್ಥಿರಾಂಕ ggml_tensor *mul = cgraph->nodes[ಗಂಟು_idx + 1];

        ಆದರೆ (!mmid_mul_ok(mmid, mul)) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }
    }

    ಹಿಂತಿರುಗಿ ನಿಜ;
}

ತಾರ್ಕಿಕ ggml_vk_can_fuse_ssm_conv(ಸ್ಥಿರಾಂಕ ggml_backend_vk_context * ctx, ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_cgraph * cgraph,
                                      ಪೂರ್ಣಾಂಕ ಗಂಟು_idx, ಪೂರ್ಣಾಂಕ ಅಂಕೆ_extra) {
    ಸ್ಥಿರಾಂಕ ggml_tensor * conv = cgraph->nodes[ಗಂಟು_idx];
    ಆದರೆ (conv->op != GGML_OP_SSM_CONV) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಸ್ಥಿರಾಂಕ ggml_tensor * silu = ಶೂನ್ಯಸೂಚಕ;
    ಸ್ಥಿರಾಂಕ ggml_tensor * bias = ಶೂನ್ಯಸೂಚಕ;

    ಆದರೆ (ಅಂಕೆ_extra == 1) {
        ಆದರೆ (!ggml_can_fuse(cgraph, ಗಂಟು_idx, { GGML_OP_SSM_CONV, GGML_OP_UNARY })) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }
        silu = cgraph->nodes[ಗಂಟು_idx + 1];
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (ಅಂಕೆ_extra == 2) {
        ಆದರೆ (!ggml_can_fuse(cgraph, ಗಂಟು_idx, { GGML_OP_SSM_CONV, GGML_OP_ಕೂಡಿಸು, GGML_OP_UNARY })) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }
        ಸ್ಥಿರಾಂಕ ggml_tensor * ಕೂಡಿಸು_2 = cgraph->nodes[ಗಂಟು_idx + 1];
        silu = cgraph->nodes[ಗಂಟು_idx + 2];
        bias = (ಕೂಡಿಸು_2->src[0] == conv) ? ಕೂಡಿಸು_2->src[1] : ಕೂಡಿಸು_2->src[0];

        ಆದರೆ (bias->type != GGML_TYPE_F32 || !ggml_is_contiguous(bias)) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }
        // bias must be channel-wise (one element per channel of the conv output)
        ಆದರೆ (ggml_nelements(bias) != conv->ne[0] || bias->ne[0] != conv->ne[0]) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }
        ಆದರೆ (ಕೂಡಿಸು_2->type != GGML_TYPE_F32) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }
        // The shader doesn't apply per-tensor offsets, so reject misaligned bias.
        ಆದರೆ (get_misalign_ಬೈಟುಗಳು(ctx, bias) != 0) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }
    } ಇಲ್ಲದಿದ್ದರೆ {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಆದರೆ (ggml_get_unary_op(silu) != GGML_UNARY_OP_SILU) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
    ಆದರೆ (conv->type != GGML_TYPE_F32 || silu->type != GGML_TYPE_F32) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
    // The shader writes to the fused dst using its own strides, but the push constants don't
    // carry a per-tensor offset, so the binding must be naturally aligned.
    ಆದರೆ (get_misalign_ಬೈಟುಗಳು(ctx, silu) != 0) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
    ಹಿಂತಿರುಗಿ ನಿಜ;
}

ತಾರ್ಕಿಕ ggml_vk_can_fuse_topk_moe(ggml_backend_vk_context * ctx, ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_cgraph * cgraph,
                                      ಪೂರ್ಣಾಂಕ ಗಂಟು_idx, topk_moe_ಬಹುಲಕ mode) {

    ಸ್ಥಿರಾಂಕ ggml_tensor * softmax;
    ಸ್ಥಿರಾಂಕ ggml_tensor * weights;
    ಸ್ಥಿರಾಂಕ ggml_tensor * get_ಅಡ್ಡಸಾಲುಗಳು;
    ಸ್ಥಿರಾಂಕ ggml_tensor * argsort;

    ಆಯ್ಕೆ (mode) {
    ಸ್ಥಿತಿ TOPK_MOE_EARLY_SOFTMAX_NORM:
        softmax = cgraph->nodes[ಗಂಟು_idx + 0];
        weights = cgraph->nodes[ಗಂಟು_idx + 9];
        get_ಅಡ್ಡಸಾಲುಗಳು = cgraph->nodes[ಗಂಟು_idx + 4];
        argsort = cgraph->nodes[ಗಂಟು_idx + 2];
        ನಿಲ್ಲಿಸು;
    ಸ್ಥಿತಿ TOPK_MOE_SIGMOID_NORM_BIAS:
        softmax = cgraph->nodes[ಗಂಟು_idx + 0]; // really sigmoid
        weights = cgraph->nodes[ಗಂಟು_idx + 10];
        get_ಅಡ್ಡಸಾಲುಗಳು = cgraph->nodes[ಗಂಟು_idx + 5];
        argsort = cgraph->nodes[ಗಂಟು_idx + 3];
        ಆದರೆ (ggml_get_unary_op(softmax) != GGML_UNARY_OP_SIGMOID) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }
        // bias is expected to be 1D
        ಆದರೆ (ggml_nrows(cgraph->nodes[ಗಂಟು_idx + 2]->src[1]) != 1 ||
            !ggml_is_contiguous(cgraph->nodes[ಗಂಟು_idx + 2]->src[1])) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }
        // sigmoid fusion seems to generate infinities on moltenvk
        ಆದರೆ (ctx->device->driver_id == vk::DriverId::eMoltenvk) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }
        ನಿಲ್ಲಿಸು;
    ಸ್ಥಿತಿ TOPK_MOE_SQRT_SOFTPLUS_NORM_BIAS:
        softmax = cgraph->nodes[ಗಂಟು_idx + 0]; // really softplus
        weights = cgraph->nodes[ಗಂಟು_idx + 11];
        get_ಅಡ್ಡಸಾಲುಗಳು = cgraph->nodes[ಗಂಟು_idx + 6];
        argsort = cgraph->nodes[ಗಂಟು_idx + 4];
        ಆದರೆ (ggml_get_unary_op(softmax) != GGML_UNARY_OP_SOFTPLUS) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }
        // bias is expected to be 1D
        ಆದರೆ (ggml_nrows(cgraph->nodes[ಗಂಟು_idx + 3]->src[1]) != 1 ||
            !ggml_is_contiguous(cgraph->nodes[ಗಂಟು_idx + 3]->src[1])) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }
        ನಿಲ್ಲಿಸು;
    ಸ್ಥಿತಿ TOPK_MOE_EARLY_SOFTMAX:
        softmax = cgraph->nodes[ಗಂಟು_idx + 0];
        weights = cgraph->nodes[ಗಂಟು_idx + 4];
        get_ಅಡ್ಡಸಾಲುಗಳು = cgraph->nodes[ಗಂಟು_idx + 4];
        argsort = cgraph->nodes[ಗಂಟು_idx + 2];
        ನಿಲ್ಲಿಸು;
    ಸ್ಥಿತಿ TOPK_MOE_LATE_SOFTMAX:
        softmax = cgraph->nodes[ಗಂಟು_idx + 4];
        weights = cgraph->nodes[ಗಂಟು_idx + 5];
        get_ಅಡ್ಡಸಾಲುಗಳು = cgraph->nodes[ಗಂಟು_idx + 2];
        argsort = cgraph->nodes[ಗಂಟು_idx + 0];
        ನಿಲ್ಲಿಸು;
    ಪೂರ್ವನಿಯೋಜಿತ:
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ggml_tensor * probs = get_ಅಡ್ಡಸಾಲುಗಳು->src[0];
    ಆದರೆ (probs->op != GGML_OP_RESHAPE) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
    probs = probs->src[0];
    ggml_tensor * selection_probs = argsort->src[0];

    ಆದರೆ (probs != selection_probs &&
        mode != TOPK_MOE_SIGMOID_NORM_BIAS &&
        mode != TOPK_MOE_SQRT_SOFTPLUS_NORM_BIAS) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಆದರೆ (!ggml_is_contiguous(softmax->src[0]) || !ggml_is_contiguous(weights)) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಆದರೆ (softmax->op == GGML_OP_SOFT_MAX) {
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * op_params = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *)softmax->op_params;

        ದಶಮಾಂಶ scale = op_params[0];
        ದಶಮಾಂಶ max_bias = op_params[1];

        ಆದರೆ (scale != 1.0f || max_bias != 0.0f) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }

        // don't fuse when masks or sinks are present
        ಆದರೆ (softmax->src[1] || softmax->src[2]) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }
    }

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ n_expert = softmax->ne[0];
    ಆದರೆ (n_expert > (1 << (ಅಂಕೆ_topk_moe_pipelines-1))) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಆದರೆ (!ctx->device->subgroup_arithmetic ||
        !ctx->device->subgroup_shuffle ||
        !ctx->device->subgroup_require_ತುಂಬಿದ_support ||
        ctx->device->disable_fusion) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಹಿಂತಿರುಗಿ ನಿಜ;
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_vk_ಹೊಂದಿಕೆ_ops(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_cgraph * cgraph, ಪೂರ್ಣಾಂಕ ಗಂಟು_idx,
                              ಸ್ಥಿರಾಂಕ std::initializer_list<ggml_op> & ops) {
    ಆದರೆ (ಗಂಟು_idx + (ಪೂರ್ಣಾಂಕ) ops.size() > cgraph->n_nodes) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
    ಗಾಗಿ (size_t j = 0; j < ops.size(); ++j) {
        ಸ್ಥಿರಾಂಕ ggml_tensor * ಗಂಟು = cgraph->nodes[ಗಂಟು_idx + j];
        ಆದರೆ (ಗಂಟು->op != ops.begin()[j] ||
            (ಗಂಟು->flags & GGML_TENSOR_FLAG_COMPUTE) == 0 ||
            (ಗಂಟು->flags & GGML_TENSOR_FLAG_OUTPUT) != 0) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }
    }
    ಹಿಂತಿರುಗಿ ನಿಜ;
}

ತಾರ್ಕಿಕ ggml_vk_can_fuse_topk_qsa(ggml_backend_vk_context * ctx, ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_cgraph * cgraph, ಪೂರ್ಣಾಂಕ ಗಂಟು_idx) {
    ಆದರೆ (ctx->device->disable_fusion || !ctx->device->pipeline_topk_radix_qsa) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ n_ops = topk_qsa_ನಮೂನೆ.size();
    ಆದರೆ (!ggml_vk_ಹೊಂದಿಕೆ_ops(cgraph, ಗಂಟು_idx, topk_qsa_ನಮೂನೆ) ||
        !ggml_check_ಅಂಚುಗಳು(cgraph, ಗಂಟು_idx, topk_qsa_ಅಂಚುಗಳು)) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    // elided nodes must be single-use (cpy counts its own src[1] self-reference)
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < n_ops - 1; ++j) {
        ಸ್ಥಿರಾಂಕ ggml_tensor * ಗಂಟು = cgraph->nodes[ಗಂಟು_idx + j];
        ಸ್ಥಿರಾಂಕ int32_t want = ಗಂಟು->op == GGML_OP_CPY ? 2 : 1;
        ಆದರೆ (ggml_ಗಂಟು_get_use_ಎಣಿಕೆ(cgraph, ಗಂಟು_idx + j) != want) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }
    }

    ಸ್ಥಿರಾಂಕ ggml_tensor * get_ಅಡ್ಡಸಾಲುಗಳು = cgraph->nodes[ಗಂಟು_idx + 0];
    ಸ್ಥಿರಾಂಕ ggml_tensor * ಕೂಡಿಸು_2      = cgraph->nodes[ಗಂಟು_idx + n_ops - 2];
    ಸ್ಥಿರಾಂಕ ggml_tensor * ತುದಿ_k    = cgraph->nodes[ಗಂಟು_idx + n_ops - 1];

    ಸ್ಥಿರಾಂಕ ggml_tensor * scores   = get_ಅಡ್ಡಸಾಲುಗಳು->src[0]; // [n_tps, n_blocks, n_stream]
    ಸ್ಥಿರಾಂಕ ggml_tensor * cell_blk = get_ಅಡ್ಡಸಾಲುಗಳು->src[1]; // [n_kv, n_stream]
    ಸ್ಥಿರಾಂಕ ggml_tensor * expanded = ಕೂಡಿಸು_2->src[0];      // [n_kv, n_tps, n_stream]

    // raw mask: follow the reshape/cpy chain back to the materialized f16 input
    ಸ್ಥಿರಾಂಕ ggml_tensor * ಮುಖವಾಡ = ಕೂಡಿಸು_2->src[1];
    ಆಗ (ಮುಖವಾಡ && (ಮುಖವಾಡ->op == GGML_OP_RESHAPE || ಮುಖವಾಡ->op == GGML_OP_CPY)) {
        ಮುಖವಾಡ = ಮುಖವಾಡ->src[0];
    }
    ಆದರೆ (!ಮುಖವಾಡ || ಮುಖವಾಡ->type != GGML_TYPE_F16) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಆದರೆ (scores->type != GGML_TYPE_F32 || cell_blk->type != GGML_TYPE_I32 || ತುದಿ_k->type != GGML_TYPE_I32) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
    ಆದರೆ (!ggml_is_contiguous(scores) || !ggml_is_contiguous(cell_blk) || !ggml_is_contiguous(ಮುಖವಾಡ) ||
        !ggml_is_contiguous(expanded) || !ggml_is_contiguous(ತುದಿ_k)) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಸ್ಥಿರಾಂಕ int64_t n_tps    = scores->ne[0];
    ಸ್ಥಿರಾಂಕ int64_t n_ಖಂಡಗಳು = scores->ne[1];
    ಸ್ಥಿರಾಂಕ int64_t n_ಹರಿವು = scores->ne[2];
    ಸ್ಥಿರಾಂಕ int64_t n_kv     = cell_blk->ne[0];
    ಸ್ಥಿರಾಂಕ int64_t ಅಗಲ    = ತುದಿ_k->ne[0];

    // pin the indexer layout the shader's addressing assumes
    ಆದರೆ (scores->ne[3] != 1 || cell_blk->ne[1] != n_ಹರಿವು || ggml_nrows(cell_blk) != n_ಹರಿವು ||
        ggml_nelements(ಮುಖವಾಡ) != n_kv * n_tps * n_ಹರಿವು ||
        expanded->ne[0] != n_kv || expanded->ne[1] != n_tps || expanded->ne[2] != n_ಹರಿವು ||
        ತುದಿ_k->ne[1] != n_tps || ತುದಿ_k->ne[2] != n_ಹರಿವು || ತುದಿ_k->ne[3] != 1 ||
        n_ಖಂಡಗಳು <= 0 || n_kv <= 0 || ಅಗಲ <= 0 || ಅಗಲ > n_kv) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    // only worth it in the radix regime; small k uses the faster tournament unfused
    ಸ್ಥಿರಾಂಕ uint32_t k_min_pipeline = std::max((uint32_t) log2f(ದಶಮಾಂಶ(ಅಗಲ)) + 1, ctx->device->subgroup_ಗಾತ್ರ_log2);
    ಆದರೆ (k_min_pipeline < ಅಂಕೆ_topk_pipelines && ctx->device->pipeline_topk_f32[k_min_pipeline]) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
    ಹಿಂತಿರುಗಿ ನಿಜ;
}

ತಾರ್ಕಿಕ ggml_vk_can_fuse_rope_ಗಣ_ಅಡ್ಡಸಾಲುಗಳು(ggml_backend_vk_context * ctx, ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_cgraph * cgraph,
                                           ಪೂರ್ಣಾಂಕ ಗಂಟು_idx) {
    ಸ್ಥಿರಾಂಕ ggml_tensor *rope = cgraph->nodes[ಗಂಟು_idx + 0];
    ಸ್ಥಿರಾಂಕ ggml_tensor *view = cgraph->nodes[ಗಂಟು_idx + 1];
    ಸ್ಥಿರಾಂಕ ggml_tensor *ಗಣ_ಅಡ್ಡಸಾಲುಗಳು = cgraph->nodes[ಗಂಟು_idx + 2];

    // The set_rows epilogue uses one index per ne2 slice and does not encode ne3.
    ಆದರೆ (rope->src[0]->ne[3] != 1) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಆದರೆ (ಗಣ_ಅಡ್ಡಸಾಲುಗಳು->type != GGML_TYPE_F32 && ಗಣ_ಅಡ್ಡಸಾಲುಗಳು->type != GGML_TYPE_F16) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    // The shader reads each aligned I64 index as a uvec2 and uses its low 32 bits.
    ಆದರೆ (ಗಣ_ಅಡ್ಡಸಾಲುಗಳು->src[1]->type != GGML_TYPE_I64 || !ggml_is_contiguous(ಗಣ_ಅಡ್ಡಸಾಲುಗಳು->src[1]) ||
        ಗಣ_ಅಡ್ಡಸಾಲುಗಳು->nb[0] != ggml_type_ಗಾತ್ರ(ಗಣ_ಅಡ್ಡಸಾಲುಗಳು->type) || get_misalign_ಬೈಟುಗಳು(ctx, ಗಣ_ಅಡ್ಡಸಾಲುಗಳು->src[1]) != 0) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    // SET_ROWS consumes one flattened [ne0*ne1] row for each ne2 slice.
    ಆದರೆ (!ggml_is_contiguous(view) ||
        view->ne[0] != rope->ne[0] * rope->ne[1] || view->ne[1] != rope->ne[2] ||
        view->ne[2] != 1 || view->ne[3] != 1 ||
        ggml_nelements(ಗಣ_ಅಡ್ಡಸಾಲುಗಳು->src[1]) != rope->ne[2]) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    // Only norm/neox/mrope/imrope shaders have the fusion code
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ mode = ((ಸ್ಥಿರಾಂಕ int32_t *) rope->op_params)[2];
    ಆದರೆ (mode != GGML_ROPE_TYPE_ಸಾಮಾನ್ಯ && mode != GGML_ROPE_TYPE_NEOX &&
        mode != GGML_ROPE_TYPE_MROPE && mode != GGML_ROPE_TYPE_IMROPE) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಹಿಂತಿರುಗಿ ನಿಜ;
}

ತಾರ್ಕಿಕ ggml_vk_can_fuse_rms_norm_ಗಣ_ಅಡ್ಡಸಾಲುಗಳು(ggml_backend_vk_context * ctx, ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_cgraph * cgraph,
                                               ಪೂರ್ಣಾಂಕ ಗಂಟು_idx) {
    ಸ್ಥಿರಾಂಕ ggml_tensor * rms = cgraph->nodes[ಗಂಟು_idx];
    ಸ್ಥಿರಾಂಕ ggml_tensor * view = cgraph->nodes[ಗಂಟು_idx + 1];
    ಸ್ಥಿರಾಂಕ ggml_tensor * ಗಣ_ಅಡ್ಡಸಾಲುಗಳು = cgraph->nodes[ಗಂಟು_idx + 2];

    // The RMS kernel reads F32 and writes directly to the F32 or F16 SET_ROWS destination.
    ಆದರೆ (rms->src[0]->type != GGML_TYPE_F32 || rms->type != GGML_TYPE_F32 ||
        (ಗಣ_ಅಡ್ಡಸಾಲುಗಳು->type != GGML_TYPE_F32 && ಗಣ_ಅಡ್ಡಸಾಲುಗಳು->type != GGML_TYPE_F16) ||
        ಗಣ_ಅಡ್ಡಸಾಲುಗಳು->src[1]->type != GGML_TYPE_I64 || !ggml_is_contiguous(ಗಣ_ಅಡ್ಡಸಾಲುಗಳು->src[1]) ||
        ಗಣ_ಅಡ್ಡಸಾಲುಗಳು->nb[0] != ggml_type_ಗಾತ್ರ(ಗಣ_ಅಡ್ಡಸಾಲುಗಳು->type) || get_misalign_ಬೈಟುಗಳು(ctx, ಗಣ_ಅಡ್ಡಸಾಲುಗಳು->src[1]) != 0) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
    // As with the ROPE epilogue, each ne2 slice supplies one flattened row and ne3 is not encoded.
    ಆದರೆ (rms->ne[3] != 1 || !ggml_is_contiguous(rms->src[0]) || !ggml_is_contiguous(view)) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
    ಆದರೆ (view->ne[0] != rms->ne[0] * rms->ne[1] || view->ne[1] != rms->ne[2] ||
        view->ne[2] != 1 || view->ne[3] != 1 ||
        ggml_nelements(ಗಣ_ಅಡ್ಡಸಾಲುಗಳು->src[1]) != rms->ne[2]) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಹಿಂತಿರುಗಿ ನಿಜ;
}

ತಾರ್ಕಿಕ ggml_vk_can_fuse_snake(ggml_backend_vk_context * ctx, ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_cgraph * cgraph, ಪೂರ್ಣಾಂಕ ಗಂಟು_idx) {
    GGML_ಬಳಸದ(ctx);
    ಆದರೆ (!ggml_can_fuse(cgraph, ಗಂಟು_idx, snake_ನಮೂನೆ)) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಸ್ಥಿರಾಂಕ ggml_tensor * mul0     = cgraph->nodes[ಗಂಟು_idx + 0];
    ಸ್ಥಿರಾಂಕ ggml_tensor * sin_ಗಂಟು = cgraph->nodes[ಗಂಟು_idx + 1];
    ಸ್ಥಿರಾಂಕ ggml_tensor * sqr      = cgraph->nodes[ಗಂಟು_idx + 2];
    ಸ್ಥಿರಾಂಕ ggml_tensor * mul1     = cgraph->nodes[ಗಂಟು_idx + 3];
    ಸ್ಥಿರಾಂಕ ggml_tensor * ಕೂಡಿಸು_2      = cgraph->nodes[ಗಂಟು_idx + 4];

    ಸ್ಥಿರಾಂಕ ggml_tensor * x = ggml_are_same_shape(mul0, mul0->src[0]) ? mul0->src[0] : mul0->src[1];
    ಸ್ಥಿರಾಂಕ ggml_tensor * a = (x == mul0->src[0]) ? mul0->src[1] : mul0->src[0];

    ಸ್ಥಿರಾಂಕ ggml_tensor * inv_b    = (mul1->src[0] == sqr) ? mul1->src[1] : mul1->src[0];
    ಸ್ಥಿರಾಂಕ ggml_tensor * x_ಒಳ_ಕೂಡಿಸು = (ಕೂಡಿಸು_2->src[0] == mul1) ? ಕೂಡಿಸು_2->src[1] : ಕೂಡಿಸು_2->src[0];

    ಆದರೆ (x_ಒಳ_ಕೂಡಿಸು != x) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
    ಆದರೆ (x->type != GGML_TYPE_F32 && x->type != GGML_TYPE_F16 && x->type != GGML_TYPE_BF16) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
    // Shader bindings: data_a is A_TYPE so it follows x's precision, while
    // data_b and data_c are hardcoded float, so the broadcast operands must
    // be F32 regardless of x's type.
    ಆದರೆ (a->type     != GGML_TYPE_F32) ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    ಆದರೆ (inv_b->type != GGML_TYPE_F32) ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    // Chain intermediates and output share x's precision (single A_TYPE / D_TYPE pipeline).
    ಆದರೆ (mul0->type     != x->type) ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    ಆದರೆ (sin_ಗಂಟು->type != x->type) ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    ಆದರೆ (sqr->type      != x->type) ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    ಆದರೆ (mul1->type     != x->type) ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    ಆದರೆ (ಕೂಡಿಸು_2->type      != x->type) ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    ಆದರೆ (!ggml_are_same_shape(a, inv_b)) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
    ಆದರೆ (a->ne[0] != 1 || a->ne[1] != x->ne[1]) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
    // Dispatch is 2D over (ne0, ne1), so x and add must be 2D and a / inv_b
    // must collapse to [1, C, 1, 1]. Higher dims are not handled by the shader.
    ಆದರೆ (x->ne[2]     != 1 || x->ne[3]     != 1) ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    ಆದರೆ (ಕೂಡಿಸು_2->ne[2]   != 1 || ಕೂಡಿಸು_2->ne[3]   != 1) ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    ಆದರೆ (a->ne[2]     != 1 || a->ne[3]     != 1) ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    ಆದರೆ (inv_b->ne[2] != 1 || inv_b->ne[3] != 1) ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    // Shader uses idx = i0 + i1 * ne0 and reads data_b[i1] / data_c[i1],
    // so every operand must be contiguous.
    ಆದರೆ (!ggml_is_contiguous(x) || !ggml_is_contiguous(ಕೂಡಿಸು_2) ||
        !ggml_is_contiguous(a) || !ggml_is_contiguous(inv_b)) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
    ಹಿಂತಿರುಗಿ ನಿಜ;
}

ತಾರ್ಕಿಕ ggml_vk_tensors_overlap(ಸ್ಥಿರಾಂಕ ggml_tensor * a, ಸ್ಥಿರಾಂಕ ggml_tensor * b, ತಾರ್ಕಿಕ elementwise) {
    ggml_backend_vk_ಬಫರ್_context * a_buf_ctx = (ggml_backend_vk_ಬಫರ್_context *)a->ಬಫರ್->context;
    vk_ಬಫರ್ a_buf = a_buf_ctx->dev_ಬಫರ್;
    ggml_backend_vk_ಬಫರ್_context * b_buf_ctx = (ggml_backend_vk_ಬಫರ್_context *)b->ಬಫರ್->context;
    vk_ಬಫರ್ b_buf = b_buf_ctx->dev_ಬಫರ್;
    ಆದರೆ (a_buf == b_buf) {
        ಸ್ವಯಂಚಾಲಿತ a_ತಳ = vk_tensor_ಸ್ಥಾನಾಂತರ(a) + a->view_offs;
        ಸ್ವಯಂಚಾಲಿತ a_ಗಾತ್ರ = ggml_nbytes(a);
        ಸ್ವಯಂಚಾಲಿತ b_ತಳ = vk_tensor_ಸ್ಥಾನಾಂತರ(b) + b->view_offs;
        ಸ್ವಯಂಚಾಲಿತ b_ಗಾತ್ರ = ggml_nbytes(b);

        ಆದರೆ (elementwise && a_ತಳ == b_ತಳ && a_ಗಾತ್ರ == b_ಗಾತ್ರ) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }

        ಆದರೆ ((b_ತಳ <= a_ತಳ && a_ತಳ < b_ತಳ + b_ಗಾತ್ರ) ||
            (a_ತಳ <= b_ತಳ && b_ತಳ < a_ತಳ + a_ಗಾತ್ರ)) {
            ಹಿಂತಿರುಗಿ ನಿಜ;
        }
    }
    ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
}

ತಾರ್ಕಿಕ ggml_vk_can_fuse_rms_norm_mul_rope(ggml_backend_vk_context * ctx, ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_cgraph * cgraph,
                                               ಪೂರ್ಣಾಂಕ ಗಂಟು_idx) {
    ಸ್ಥಿರಾಂಕ ggml_tensor *rms = cgraph->nodes[ಗಂಟು_idx + 0];
    ಸ್ಥಿರಾಂಕ ggml_tensor *mul = cgraph->nodes[ಗಂಟು_idx + 1];
    ಸ್ಥಿರಾಂಕ ggml_tensor *rope = cgraph->nodes[ಗಂಟು_idx + 2];

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ mode = ((ಸ್ಥಿರಾಂಕ int32_t *) rope->op_params)[2];

    // noncontig tensors aren't tested, and don't seem common in practice
    ಆದರೆ (!ggml_is_contiguous(rms) ||
        !ggml_is_contiguous(mul) ||
        !ggml_is_contiguous(rope)) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    // only norm/neox are handled in the shader
    ಆದರೆ (mode != GGML_ROPE_TYPE_NEOX && mode != GGML_ROPE_TYPE_ಸಾಮಾನ್ಯ) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    // shared memory size for passing data from mul->rope
    ಆದರೆ (mul->ne[0] > 1024) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    // conditions for pipeline creation
    ಆದರೆ (ಗಾತ್ರ(vk_op_rms_norm_mul_rope_ತಳ್ಳು_constants) > ctx->device->properties.limits.max_ತಳ್ಳು_Constants_ಗಾತ್ರ) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಹಿಂತಿರುಗಿ ನಿಜ;
}

uint32_t ggml_vk_fuse_multi_ಕೂಡಿಸು(ggml_backend_vk_context * ctx, ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_cgraph * cgraph, ಪೂರ್ಣಾಂಕ ಗಂಟು_idx) {

    ಸ್ಥಿರಾಂಕ ggml_tensor *ಮೊದಲ_ಗಂಟು = cgraph->nodes[ಗಂಟು_idx];
    ಆದರೆ (ಮೊದಲ_ಗಂಟು->op != GGML_OP_ಕೂಡಿಸು) {
        ಹಿಂತಿರುಗಿ 0;
    }

    ಆದರೆ (!ctx->device->multi_ಕೂಡಿಸು) {
        ಹಿಂತಿರುಗಿ 0;
    }

    int32_t ಅಂಕೆ_adds = 1;
    ಆಗ (ಗಂಟು_idx + ಅಂಕೆ_adds < cgraph->n_nodes &&
           cgraph->nodes[ಗಂಟು_idx + ಅಂಕೆ_adds]->op == GGML_OP_ಕೂಡಿಸು &&
           ಅಂಕೆ_adds < MAX_FUSED_ADDS) {
        ಅಂಕೆ_adds++;
    }

    // The shader currently requires same shapes (but different strides are allowed),
    // everything f32, and no misalignment
    ಗಾಗಿ (int32_t i = 0; i < ಅಂಕೆ_adds; ++i) {
        ಸ್ಥಿರಾಂಕ ggml_tensor *ಮುಂದಿನ_ಗಂಟು = cgraph->nodes[ಗಂಟು_idx + i];
        ಆದರೆ (!ggml_are_same_shape(ಮೊದಲ_ಗಂಟು, ಮುಂದಿನ_ಗಂಟು->src[0]) ||
            !ggml_are_same_shape(ಮೊದಲ_ಗಂಟು, ಮುಂದಿನ_ಗಂಟು->src[1]) ||
            ಮುಂದಿನ_ಗಂಟು->type != GGML_TYPE_F32 ||
            ಮುಂದಿನ_ಗಂಟು->src[0]->type != GGML_TYPE_F32 ||
            ಮುಂದಿನ_ಗಂಟು->src[1]->type != GGML_TYPE_F32 ||
            get_misalign_ಬೈಟುಗಳು(ctx, ಮುಂದಿನ_ಗಂಟು) ||
            get_misalign_ಬೈಟುಗಳು(ctx, ಮುಂದಿನ_ಗಂಟು->src[0]) ||
            get_misalign_ಬೈಟುಗಳು(ctx, ಮುಂದಿನ_ಗಂಟು->src[1])) {
            ಅಂಕೆ_adds = i;
        }
    }

    // Verify we can fuse these
    ggml_op adds[MAX_FUSED_ADDS];
    ಗಾಗಿ (int32_t i = 0; i < ಅಂಕೆ_adds; ++i) {
        adds[i] = GGML_OP_ಕೂಡಿಸು;
    }

    // decrease num_adds if they can't all be fused
    ಆಗ (ಅಂಕೆ_adds > 1 && !ggml_can_fuse(cgraph, ಗಂಟು_idx, adds, ಅಂಕೆ_adds)) {
        ಅಂಕೆ_adds--;
    }

    // a single add is not "fused", so just return zero
    ಆದರೆ (ಅಂಕೆ_adds == 1) {
        ಹಿಂತಿರುಗಿ 0;
    }
    ಹಿಂತಿರುಗಿ ಅಂಕೆ_adds;
}

ಸ್ಥಿರ int32_t ಕಂಡುಹಿಡಿ_ಮೊದಲ_ಗಣ(uint32_t x) {
    int32_t ret = 0;
    ಆದರೆ (!x) {
        ಹಿಂತಿರುಗಿ -1;
    }
    ಆಗ (!(x & 1)) {
        x >>= 1;
        ret++;
    }
    ಹಿಂತಿರುಗಿ ret;
}

ಸ್ಥಿರ ggml_ಸ್ಥಿತಿ ggml_backend_vk_ನಕ್ಷೆ_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    VK_LOG_DEBUG("ggml_backend_vk_graph_compute(" << cgraph->n_nodes << " nodes)");
    ggml_backend_vk_context * ctx = (ggml_backend_vk_context *)backend->context;

    ctx->device->diag_cgraph = ಶೂನ್ಯಸೂಚಕ;
    ctx->device->diag_prev_ಆರಂಭ = -1;
    ctx->device->diag_prev_ಕೊನೆ = -1;

    ಆದರೆ (vk_instance.debug_utils_support) {
        ctx->device->debug_cmdbuf_idx = 0;
    }

    // queue scope, so it encloses every submit this evaluation makes.
    // closed when the function returns
    ggml_vk_debug_label ಸರತಿ_dbg(ctx->device->compute_ಸರತಿ->handle.get(), "ggml_backend_vk_graph_compute");

    ctx->prealloc_ಗಾತ್ರ_ಕೂಡಿಸು_rms_partials_ಸ್ಥಾನಾಂತರ = 0;
    ctx->do_ಕೂಡಿಸು_rms_partials = ಸುಳ್ಳು;
    ctx->do_ಕೂಡಿಸು_rms_partials_ಸ್ಥಾನಾಂತರ_calculation = ಸುಳ್ಳು;

    ಪೂರ್ಣಾಂಕ ಕೊನೆಯ_ಗಂಟು = cgraph->n_nodes - 1;

    // If the last op in the cgraph isn't backend GPU, the command buffer doesn't get closed properly
    ಆಗ (ಕೊನೆಯ_ಗಂಟು > 0 && (ggml_vk_is_ಖಾಲಿಗಳು(cgraph->nodes[ಕೊನೆಯ_ಗಂಟು]) || ((cgraph->nodes[ಕೊನೆಯ_ಗಂಟು]->flags & GGML_TENSOR_FLAG_COMPUTE) == 0))) {
        ಕೊನೆಯ_ಗಂಟು -= 1;
    }

    // Reserve tensor context space for all nodes
    ctx->tensor_ctxs.resize(cgraph->n_nodes);

    ತಾರ್ಕಿಕ ಮೊದಲ_ಗಂಟು_ಒಳ_batch = ನಿಜ; // true if next node will be first node in a batch
    ಪೂರ್ಣಾಂಕ submit_ಗಂಟು_idx = 0; // index to first node in a batch

    ggml_vk_submit_transfer_ctx(ctx);

    vk_context compute_ctx;
    ಆದರೆ (vk_perf_logger_enabled) {
        // allocate/resize the query pool
        ಆದರೆ (ctx->ಅಂಕೆ_queries < cgraph->n_nodes + 1) {
            ಆದರೆ (ctx->query_pool) {
                ctx->device->device.destroyQueryPool(ctx->query_pool);
            }
            vk::QueryPoolCreateInfo query_create_info;
            query_create_info.queryType = vk::QueryType::eTimestamp;
            query_create_info.query_ಎಣಿಕೆ = cgraph->n_nodes + 100;
            ctx->query_pool = ctx->device->device.createQueryPool(query_create_info);
            ctx->ಅಂಕೆ_queries = query_create_info.query_ಎಣಿಕೆ;
            ctx->query_fusion_ಹೆಸರುಗಳು.resize(ctx->ಅಂಕೆ_queries);
            ctx->query_fusion_ಗಂಟು_ಎಣಿಕೆ.resize(ctx->ಅಂಕೆ_queries);
            ctx->query_nodes.resize(ctx->ಅಂಕೆ_queries);
            ctx->query_ಗಂಟು_idx.resize(ctx->ಅಂಕೆ_queries);
        }

        ctx->device->device.resetQueryPool(ctx->query_pool, 0, cgraph->n_nodes+1);
        std::fill(ctx->query_fusion_ಹೆಸರುಗಳು.begin(), ctx->query_fusion_ಹೆಸರುಗಳು.end(), ಶೂನ್ಯಸೂಚಕ);
        std::fill(ctx->query_fusion_ಗಂಟು_ಎಣಿಕೆ.begin(), ctx->query_fusion_ಗಂಟು_ಎಣಿಕೆ.end(), 0);
        std::fill(ctx->query_nodes.begin(), ctx->query_nodes.end(), ಶೂನ್ಯಸೂಚಕ);
        std::fill(ctx->query_ಗಂಟು_idx.begin(), ctx->query_ಗಂಟು_idx.end(), 0);

        GGML_ASSERT(ctx->compute_ctx.expired());
        compute_ctx = ggml_vk_get_compute_ctx(ctx);
        ctx->query_idx = 0;
        compute_ctx->s->ಬಫರ್->buf.writeTimestamp(vk::Pipeline_Stage_Flag_ಬಿಟ್ಟುಗಳು::eAllCommands, ctx->query_pool, ctx->query_idx++);
        ggml_vk_sync_buffers(ctx, compute_ctx);
    }

    ctx->prealloc_y_ಕೊನೆಯ_pipeline_ಬಳಸಿದ = ಶೂನ್ಯಸೂಚಕ;
    ctx->prealloc_y_ಕೊನೆಯ_tensor_ಬಳಸಿದ = ಶೂನ್ಯಸೂಚಕ;
    ctx->prealloc_y_ಕೊನೆಯ_k_padded = ಸುಳ್ಳು;

    ಆದರೆ (ctx->prealloc_ಗಾತ್ರ_ಕೂಡಿಸು_rms_partials) {
        ggml_vk_preallocate_buffers(ctx, ಶೂನ್ಯಸೂಚಕ);
        compute_ctx = ggml_vk_get_compute_ctx(ctx);
        // initialize partial sums to zero.
        ggml_vk_ಬಫರ್_memset_async(compute_ctx, ctx->prealloc_ಕೂಡಿಸು_rms_partials, 0, 0, ctx->prealloc_ಗಾತ್ರ_ಕೂಡಿಸು_rms_partials);
        ggml_vk_sync_buffers(ctx, compute_ctx);
    }

    // Submit after enough work has accumulated, to overlap CPU cmdbuffer generation with GPU execution.
    // Estimate the amount of compute work using flops, and submit every 200 GFLOP
    // (and scaled down based on total graph flops, so smaller models submit earlier).
    // Also submit at least every 100 nodes, in case there are workloads without heavy compute.
    uint32_t submitted_nodes = 0;
    uint32_t submit_ಎಣಿಕೆ = 0;
    uint64_t batch_flops = 0;
    uint64_t ಒಟ್ಟು_flops = 0;
    uint64_t flops_cap = 200'000'000'000ULL;

    // On weaker AMD GPUs larger submissions can hit a driver timeout, submit more often to avoid this
    ಆದರೆ (ctx->device->vendor_id == VK_VENDOR_ID_AMD && ctx->device->shader_core_ಎಣಿಕೆ > 0) {
        ಆದರೆ (ctx->device->architecture == AMD_GCN && ctx->device->shader_core_ಎಣಿಕೆ < 32) {
            flops_cap = 500'000'000ULL * ctx->device->shader_core_ಎಣಿಕೆ;
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (ctx->device->architecture != AMD_GCN && ctx->device->shader_core_ಎಣಿಕೆ < 24) {
            flops_cap = 2'000'000'000ULL * ctx->device->shader_core_ಎಣಿಕೆ;
        }
    }
    uint64_t flops_ಪ್ರತಿ_submit = std::min(flops_cap, ctx->ಕೊನೆಯ_ಒಟ್ಟು_flops / 40u);

    ಸ್ವಯಂಚಾಲಿತ ಸ್ಥಿರಾಂಕ submit_after = [&](ಪೂರ್ಣಾಂಕ start, ಪೂರ್ಣಾಂಕ end) {
        ಆದರೆ (ctx->device->serialize_submissions) {
            ಪ್ರಯತ್ನಿಸು {
                ಸ್ವಯಂಚಾಲಿತ res = ctx->device->device.waitForFences({ ctx->fence }, ನಿಜ, UINT64_MAX);
                ಆದರೆ (res != vk::ಫಲಿತಾಂಶ_2::eSuccess) {
                    GGML_LOG_ERROR("ggml_vulkan: waitForFences error during serialized submission\n");
                    ಎಸೆ vk::SystemError(vk::make_error_code(res), "ggml_vulkan: waitForFences during serialized submission");
                }
            } ಹಿಡಿ (vk::Device_ಕಳೆದುಹೋದ_Error &) {
                ggml_vk_print_device_fault_info(ctx->device);
                GGML_LOG_ERROR("ggml_vulkan: device lost on %s waiting for submission (nodes %d to %d):\n",
                        ctx->device->name.c_str(), start, end);
                ggml_vk_print_ಗಂಟು_list(cgraph, start, end);
                ಎಸೆ;
            }
            ctx->device->device.resetFences({ ctx->fence });
            ctx->submit_pending = ಸುಳ್ಳು;
            ctx->device->diag_cgraph = cgraph;
            ctx->device->diag_prev_ಆರಂಭ = start;
            ctx->device->diag_prev_ಕೊನೆ = end;
        }
        ಮೊದಲ_ಗಂಟು_ಒಳ_batch = ನಿಜ;
        submitted_nodes = 0;
        batch_flops = 0;
        ಆದರೆ (submit_ಎಣಿಕೆ < 3) {
            flops_ಪ್ರತಿ_submit *= 2;
        }
        submit_ಎಣಿಕೆ++;
    };

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < cgraph->n_nodes; i++) {
        ಆದರೆ (ಮೊದಲ_ಗಂಟು_ಒಳ_batch) {
            submit_ಗಂಟು_idx = i;
        }

        {
            ಸ್ವಯಂಚಾಲಿತ ಗಂಟು_flops = ggml_vk_get_ಗಂಟು_flops(cgraph->nodes[i]);
            ಒಟ್ಟು_flops += ಗಂಟು_flops;

            // Flush the current batch before recording a node that would push it over the flop threshold
            ಆದರೆ (flops_ಪ್ರತಿ_submit != 0 && submitted_nodes > 0 && batch_flops + ಗಂಟು_flops >= flops_ಪ್ರತಿ_submit) {
                vk_context flush_ctx = ggml_vk_get_compute_ctx(ctx);
                ggml_vk_ctx_ಕೊನೆ(flush_ctx);
                flush_ctx->exit_tensor_idx = -1;
                ctx->compute_ctx.reset();
                ggml_vk_compute_forward(ctx, cgraph, cgraph->nodes[submit_ಗಂಟು_idx], submit_ಗಂಟು_idx, ಸುಳ್ಳು);
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

        ctx->fused_topk_moe_ಬಹುಲಕ = TOPK_MOE_ಎಣಿಕೆ;
        ctx->fused_topk_moe_scale = ಸುಳ್ಳು;
        ctx->fused_topk_qsa = ಸುಳ್ಳು;
        ctx->fused_hc_post_gate = ಸುಳ್ಳು;
        ctx->fused_rms_norm_ಬಹುಲಕ = RMS_NORM_ಎಣಿಕೆ;
        ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *fusion_string {};
        ಆದರೆ (!ctx->device->disable_fusion) {
            uint32_t ಅಂಕೆ_adds = ggml_vk_fuse_multi_ಕೂಡಿಸು(ctx, cgraph, i);
            ಆದರೆ (ಅಂಕೆ_adds) {
                ctx->ಅಂಕೆ_additional_fused_ops = ಅಂಕೆ_adds - 1;
                fusion_string = "MULTI_ADD";
                std::fill_n(op_srcs_fused_elementwise, ctx->ಅಂಕೆ_additional_fused_ops + 1, ನಿಜ);
            } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (ggml_vk_can_fuse(ctx, cgraph, i, { GGML_OP_MUL_MAT, GGML_OP_ಕೂಡಿಸು, GGML_OP_ಕೂಡಿಸು })) {
                ctx->ಅಂಕೆ_additional_fused_ops = 2;
                fusion_string = "MUL_MAT_ADD_ADD";
                op_srcs_fused_elementwise[0] = ಸುಳ್ಳು;
                op_srcs_fused_elementwise[1] = ನಿಜ;
                op_srcs_fused_elementwise[2] = ನಿಜ;
            } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (ggml_vk_can_fuse(ctx, cgraph, i, { GGML_OP_MUL_MAT, GGML_OP_ಕೂಡಿಸು })) {
                ctx->ಅಂಕೆ_additional_fused_ops = 1;
                fusion_string = "MUL_MAT_ADD";
                op_srcs_fused_elementwise[0] = ಸುಳ್ಳು;
                op_srcs_fused_elementwise[1] = ನಿಜ;
            } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (ggml_vk_can_fuse(ctx, cgraph, i, { GGML_OP_MUL_MAT_ID, GGML_OP_ಕೂಡಿಸು_ID, GGML_OP_MUL })) {
                ctx->ಅಂಕೆ_additional_fused_ops = 2;
                fusion_string = "MUL_MAT_ID_ADD_ID_MUL";
                op_srcs_fused_elementwise[0] = ಸುಳ್ಳು;
                op_srcs_fused_elementwise[1] = ನಿಜ;
                op_srcs_fused_elementwise[2] = ನಿಜ;
            } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (ggml_vk_can_fuse(ctx, cgraph, i, { GGML_OP_MUL_MAT_ID, GGML_OP_ಕೂಡಿಸು_ID })) {
                ctx->ಅಂಕೆ_additional_fused_ops = 1;
                fusion_string = "MUL_MAT_ID_ADD_ID";
                op_srcs_fused_elementwise[0] = ಸುಳ್ಳು;
                op_srcs_fused_elementwise[1] = ನಿಜ;
            } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (ggml_vk_can_fuse(ctx, cgraph, i, { GGML_OP_MUL_MAT_ID, GGML_OP_MUL })) {
                ctx->ಅಂಕೆ_additional_fused_ops = 1;
                fusion_string = "MUL_MAT_ID_MUL";
                op_srcs_fused_elementwise[0] = ಸುಳ್ಳು;
                op_srcs_fused_elementwise[1] = ನಿಜ;
            } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (ggml_can_fuse_subgraph(cgraph, i, rms_norm_mul_rope_view_ಗಣ_ಅಡ್ಡಸಾಲುಗಳು_ನಮೂನೆ, { i + 4 }) &&
                       ggml_check_ಅಂಚುಗಳು(cgraph, i, rms_norm_mul_rope_view_ಗಣ_ಅಡ್ಡಸಾಲುಗಳು_ಅಂಚುಗಳು) &&
                       ggml_vk_can_fuse_rms_norm_mul_rope(ctx, cgraph, i) &&
                       ggml_vk_can_fuse_rope_ಗಣ_ಅಡ್ಡಸಾಲುಗಳು(ctx, cgraph, i + 2)) {
                ctx->ಅಂಕೆ_additional_fused_ops = 4;
                ctx->fused_rms_norm_ಬಹುಲಕ = RMS_NORM_MUL_ROPE_VIEW_ಗಣ_ಅಡ್ಡಸಾಲುಗಳು;
                fusion_string = "RMS_NORM_MUL_ROPE_VIEW_SET_ROWS";
                op_srcs_fused_elementwise[0] = ಸುಳ್ಳು;
                op_srcs_fused_elementwise[1] = ಸುಳ್ಳು;
                op_srcs_fused_elementwise[2] = ಸುಳ್ಳು;
                op_srcs_fused_elementwise[3] = ಸುಳ್ಳು;
                op_srcs_fused_elementwise[4] = ಸುಳ್ಳು;
            } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (ggml_vk_can_fuse(ctx, cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ROPE }) &&
                       ggml_vk_can_fuse_rms_norm_mul_rope(ctx, cgraph, i)) {
                ctx->ಅಂಕೆ_additional_fused_ops = 2;
                ctx->fused_rms_norm_ಬಹುಲಕ = RMS_NORM_MUL_ROPE;
                fusion_string = "RMS_NORM_MUL_ROPE";
                // rope is approximately elementwise - whole rows are done by a single workgroup and it's row-wise
                op_srcs_fused_elementwise[0] = ಸುಳ್ಳು;
                op_srcs_fused_elementwise[1] = ನಿಜ;
                op_srcs_fused_elementwise[2] = ನಿಜ;
            } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (ggml_can_fuse_subgraph(cgraph, i, hc_post_gate_ನಮೂನೆ, { i + 3 }) &&
                       ggml_check_ಅಂಚುಗಳು(cgraph, i, hc_post_gate_ಅಂಚುಗಳು) &&
                       ggml_vk_can_fuse_hc_post_gate(cgraph, i)) {
                ctx->ಅಂಕೆ_additional_fused_ops = hc_post_gate_ನಮೂನೆ.size() - 1;
                ctx->fused_hc_post_gate = ನಿಜ;
                fusion_string = "HC_POST_GATE";
                std::fill_n(op_srcs_fused_elementwise, ctx->ಅಂಕೆ_additional_fused_ops + 1, ಸುಳ್ಳು);
            } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (ggml_vk_can_fuse(ctx, cgraph, i, rms_norm_mul_ಕೂಡಿಸು_mul_ನಮೂನೆ)) {
                ctx->ಅಂಕೆ_additional_fused_ops = 3;
                ctx->fused_rms_norm_ಬಹುಲಕ = RMS_NORM_MUL_ಕೂಡಿಸು_MUL;
                fusion_string = "RMS_NORM_MUL_ADD_MUL";
                std::fill_n(op_srcs_fused_elementwise, 4, ನಿಜ);
            } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (ggml_vk_can_fuse(ctx, cgraph, i, rms_norm_mul_ಕೂಡಿಸು_ನಮೂನೆ)) {
                ctx->ಅಂಕೆ_additional_fused_ops = 2;
                ctx->fused_rms_norm_ಬಹುಲಕ = RMS_NORM_MUL_ಕೂಡಿಸು;
                fusion_string = "RMS_NORM_MUL_ADD";
                std::fill_n(op_srcs_fused_elementwise, 3, ನಿಜ);
            } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (ggml_can_fuse_subgraph(cgraph, i, rms_norm_view_ಗಣ_ಅಡ್ಡಸಾಲುಗಳು_ನಮೂನೆ, { i + 2 }) &&
                       ggml_check_ಅಂಚುಗಳು(cgraph, i, rms_norm_view_ಗಣ_ಅಡ್ಡಸಾಲುಗಳು_ಅಂಚುಗಳು) &&
                       ggml_vk_can_fuse_rms_norm_ಗಣ_ಅಡ್ಡಸಾಲುಗಳು(ctx, cgraph, i)) {
                ctx->ಅಂಕೆ_additional_fused_ops = 2;
                ctx->fused_rms_norm_ಬಹುಲಕ = RMS_NORM_VIEW_ಗಣ_ಅಡ್ಡಸಾಲುಗಳು;
                fusion_string = "RMS_NORM_VIEW_SET_ROWS";
                std::fill_n(op_srcs_fused_elementwise, 3, ಸುಳ್ಳು);
            } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (ggml_vk_can_fuse(ctx, cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL })) {
                ctx->ಅಂಕೆ_additional_fused_ops = 1;
                ctx->fused_rms_norm_ಬಹುಲಕ = RMS_NORM_MUL;
                fusion_string = "RMS_NORM_MUL";
                // rms_norm is not elementwise, but whole rows must be consumed and the scale factor computed before
                // they are overwritten, and one workgroup per row. So close enough.
                op_srcs_fused_elementwise[0] = ನಿಜ;
                op_srcs_fused_elementwise[1] = ನಿಜ;
            } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (ggml_vk_can_fuse(ctx, cgraph, i, { GGML_OP_UNARY, GGML_OP_MUL })) {
                ctx->ಅಂಕೆ_additional_fused_ops = 1;
                ಆಯ್ಕೆ (ggml_get_unary_op(cgraph->nodes[i])) {
                    ಸ್ಥಿತಿ GGML_UNARY_OP_GELU:     fusion_string = "GELU_MUL";     ನಿಲ್ಲಿಸು;
                    ಸ್ಥಿತಿ GGML_UNARY_OP_SIGMOID:  fusion_string = "SIGMOID_MUL";  ನಿಲ್ಲಿಸು;
                    ಸ್ಥಿತಿ GGML_UNARY_OP_SILU:     fusion_string = "SILU_MUL";     ನಿಲ್ಲಿಸು;
                    ಪೂರ್ವನಿಯೋಜಿತ:                     fusion_string = "SOFTPLUS_MUL"; ನಿಲ್ಲಿಸು;
                }
                op_srcs_fused_elementwise[0] = ನಿಜ;
                op_srcs_fused_elementwise[1] = ನಿಜ;
            } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (ggml_vk_can_fuse_ssm_conv(ctx, cgraph, i, 2)) {
                ctx->ಅಂಕೆ_additional_fused_ops = 2;
                fusion_string = "SSM_CONV_BIAS_SILU";
                // ssm_conv reads multiple input tokens per output, so it's not elementwise w.r.t. its srcs.
                // The downstream add and silu are elementwise on the conv output.
                op_srcs_fused_elementwise[0] = ಸುಳ್ಳು;
                op_srcs_fused_elementwise[1] = ನಿಜ;
                op_srcs_fused_elementwise[2] = ನಿಜ;
            } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (ggml_vk_can_fuse_ssm_conv(ctx, cgraph, i, 1)) {
                ctx->ಅಂಕೆ_additional_fused_ops = 1;
                fusion_string = "SSM_CONV_SILU";
                op_srcs_fused_elementwise[0] = ಸುಳ್ಳು;
                op_srcs_fused_elementwise[1] = ನಿಜ;
            } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (ggml_can_fuse_subgraph(cgraph, i, rope_view_ಗಣ_ಅಡ್ಡಸಾಲುಗಳು_ನಮೂನೆ, { i + 2 }) &&
                       ggml_check_ಅಂಚುಗಳು(cgraph, i, rope_view_ಗಣ_ಅಡ್ಡಸಾಲುಗಳು_ಅಂಚುಗಳು) &&
                       ggml_vk_can_fuse_rope_ಗಣ_ಅಡ್ಡಸಾಲುಗಳು(ctx, cgraph, i)) {
                ctx->ಅಂಕೆ_additional_fused_ops = 2;
                fusion_string = "ROPE_VIEW_SET_ROWS";
                op_srcs_fused_elementwise[0] = ಸುಳ್ಳು;
                op_srcs_fused_elementwise[1] = ಸುಳ್ಳು;
                op_srcs_fused_elementwise[2] = ಸುಳ್ಳು;
            } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (ggml_vk_can_fuse_snake(ctx, cgraph, i)) {
                ctx->ಅಂಕೆ_additional_fused_ops = 4;
                fusion_string = "SNAKE";
                // elementwise=true: snake.comp is safe under exact aliasing because each
                // thread reads data_x[idx] into a register before writing data_d[idx]
                // with a data dependency on that register. The overlap check still
                // rejects partial overlaps (different base or size).
                std::fill_n(op_srcs_fused_elementwise, 5, ನಿಜ);
            } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (ggml_vk_can_fuse_topk_qsa(ctx, cgraph, i)) {
                ctx->ಅಂಕೆ_additional_fused_ops = topk_qsa_ನಮೂನೆ.size() - 1;
                ctx->fused_topk_qsa = ನಿಜ;
                fusion_string = "TOPK_QSA";
                std::fill_n(op_srcs_fused_elementwise, ctx->ಅಂಕೆ_additional_fused_ops + 1, ಸುಳ್ಳು);
            } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (ggml_can_fuse_subgraph(cgraph, i, topk_moe_early_softmax_norm, { i + 3, i + 9 }) &&
                       ggml_check_ಅಂಚುಗಳು(cgraph, i, topk_moe_early_softmax_norm_ಅಂಚುಗಳು) &&
                       ggml_vk_can_fuse_topk_moe(ctx, cgraph, i, TOPK_MOE_EARLY_SOFTMAX_NORM)) {
                ctx->ಅಂಕೆ_additional_fused_ops = topk_moe_early_softmax_norm.size() - 1;
                // view of argsort writes to memory
                ctx->fused_ops_write_ಮುಖವಾಡ |= 1 << 3;
                ctx->fused_topk_moe_ಬಹುಲಕ = TOPK_MOE_EARLY_SOFTMAX_NORM;
                fusion_string = "TOPK_MOE_EARLY_SOFTMAX_NORM";
                std::fill_n(op_srcs_fused_elementwise, ctx->ಅಂಕೆ_additional_fused_ops + 1, ಸುಳ್ಳು);
            } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (ggml_can_fuse_subgraph(cgraph, i, topk_moe_sigmoid_norm_bias, { i + 4, i + 10 }) &&
                       ggml_check_ಅಂಚುಗಳು(cgraph, i, topk_moe_sigmoid_norm_bias_ಅಂಚುಗಳು) &&
                       ggml_vk_can_fuse_topk_moe(ctx, cgraph, i, TOPK_MOE_SIGMOID_NORM_BIAS)) {
                ctx->ಅಂಕೆ_additional_fused_ops = topk_moe_sigmoid_norm_bias.size() - 1;
                // view of argsort writes to memory
                ctx->fused_ops_write_ಮುಖವಾಡ |= 1 << 4;
                ctx->fused_topk_moe_ಬಹುಲಕ = TOPK_MOE_SIGMOID_NORM_BIAS;
                fusion_string = "TOPK_MOE_SIGMOID_NORM_BIAS";
                std::fill_n(op_srcs_fused_elementwise, ctx->ಅಂಕೆ_additional_fused_ops + 1, ಸುಳ್ಳು);
            } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (ggml_can_fuse_subgraph(cgraph, i, topk_moe_sqrt_softplus_norm_bias, { i + 5, i + 11 }) &&
                       ggml_check_ಅಂಚುಗಳು(cgraph, i, topk_moe_sqrt_softplus_norm_bias_ಅಂಚುಗಳು) &&
                       ggml_vk_can_fuse_topk_moe(ctx, cgraph, i, TOPK_MOE_SQRT_SOFTPLUS_NORM_BIAS)) {
                ctx->ಅಂಕೆ_additional_fused_ops = topk_moe_sqrt_softplus_norm_bias.size() - 1;
                // view of argsort writes to memory
                ctx->fused_ops_write_ಮುಖವಾಡ |= 1 << 5;
                ctx->fused_topk_moe_ಬಹುಲಕ = TOPK_MOE_SQRT_SOFTPLUS_NORM_BIAS;
                fusion_string = "TOPK_MOE_SQRT_SOFTPLUS_NORM_BIAS";
                std::fill_n(op_srcs_fused_elementwise, ctx->ಅಂಕೆ_additional_fused_ops + 1, ಸುಳ್ಳು);
            } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (ggml_can_fuse_subgraph(cgraph, i, topk_moe_early_softmax, { i + 3, i + 4 }) &&
                       ggml_check_ಅಂಚುಗಳು(cgraph, i, topk_moe_early_softmax_ಅಂಚುಗಳು) &&
                       ggml_vk_can_fuse_topk_moe(ctx, cgraph, i, TOPK_MOE_EARLY_SOFTMAX)) {
                ctx->ಅಂಕೆ_additional_fused_ops = topk_moe_early_softmax.size() - 1;
                // view of argsort writes to memory
                ctx->fused_ops_write_ಮುಖವಾಡ |= 1 << 3;
                ctx->fused_topk_moe_ಬಹುಲಕ = TOPK_MOE_EARLY_SOFTMAX;
                fusion_string = "TOPK_MOE_EARLY_SOFTMAX";
                std::fill_n(op_srcs_fused_elementwise, ctx->ಅಂಕೆ_additional_fused_ops + 1, ಸುಳ್ಳು);
            } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (ggml_can_fuse_subgraph(cgraph, i, topk_moe_late_softmax, { i + 1, i + 5 }) &&
                       ggml_check_ಅಂಚುಗಳು(cgraph, i, topk_moe_late_softmax_ಅಂಚುಗಳು) &&
                       ggml_vk_can_fuse_topk_moe(ctx, cgraph, i, TOPK_MOE_LATE_SOFTMAX)) {
                ctx->ಅಂಕೆ_additional_fused_ops = topk_moe_late_softmax.size() - 1;
                // view of argsort writes to memory
                ctx->fused_ops_write_ಮುಖವಾಡ |= 1 << 1;
                ctx->fused_topk_moe_ಬಹುಲಕ = TOPK_MOE_LATE_SOFTMAX;
                fusion_string = "TOPK_MOE_LATE_SOFTMAX";
                std::fill_n(op_srcs_fused_elementwise, ctx->ಅಂಕೆ_additional_fused_ops + 1, ಸುಳ್ಳು);
            }
            ಆದರೆ (ctx->fused_topk_moe_ಬಹುಲಕ != TOPK_MOE_ಎಣಿಕೆ) {
                // Look for an additional scale op to fuse - occurs in deepseek2 and nemotron3 nano.
                ಆದರೆ (ggml_can_fuse_subgraph(cgraph, i + ctx->ಅಂಕೆ_additional_fused_ops - 1, { GGML_OP_DIV, GGML_OP_RESHAPE, GGML_OP_SCALE }, { i + ctx->ಅಂಕೆ_additional_fused_ops + 1 }) ||
                    ggml_can_fuse_subgraph(cgraph, i + ctx->ಅಂಕೆ_additional_fused_ops, { GGML_OP_GET_ಅಡ್ಡಸಾಲುಗಳು, GGML_OP_SCALE }, { i + ctx->ಅಂಕೆ_additional_fused_ops + 1 })) {
                    ctx->fused_topk_moe_scale = ನಿಜ;
                    ctx->ಅಂಕೆ_additional_fused_ops++;
                    op_srcs_fused_elementwise[ctx->ಅಂಕೆ_additional_fused_ops] = ಸುಳ್ಳು;
                }
            }
        }
        GGML_ASSERT(ctx->ಅಂಕೆ_additional_fused_ops < (ಪೂರ್ಣಾಂಕ)(ಗಾತ್ರ(op_srcs_fused_elementwise) / ಗಾತ್ರ(op_srcs_fused_elementwise[0])));
        ctx->fused_ops_write_ಮುಖವಾಡ |= 1 << ctx->ಅಂಕೆ_additional_fused_ops;

        // Check whether fusion would overwrite src operands while they're still in use.
        // If so, disable fusion.
        ಆದರೆ (ctx->ಅಂಕೆ_additional_fused_ops) {
            // There are up to two output nodes - topk_moe has two.
            uint32_t ಬಿಟ್ಟುಗಳು = ctx->fused_ops_write_ಮುಖವಾಡ & ~(1 << ctx->ಅಂಕೆ_additional_fused_ops);
            ggml_tensor *output_nodes[2] {};
            output_nodes[0] = cgraph->nodes[i + ctx->ಅಂಕೆ_additional_fused_ops];
            ಆದರೆ (ಬಿಟ್ಟುಗಳು) {
                ಪೂರ್ಣಾಂಕ output_idx = ಕಂಡುಹಿಡಿ_ಮೊದಲ_ಗಣ(ಬಿಟ್ಟುಗಳು);
                GGML_ASSERT(ಬಿಟ್ಟುಗಳು == (1u << output_idx));
                output_nodes[1] = cgraph->nodes[i + output_idx];
            }

            ತಾರ್ಕಿಕ need_disable = ಸುಳ್ಳು;

            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < 2; ++j) {
                ggml_tensor *dst = output_nodes[j];
                ಆದರೆ (!dst) {
                    ಮುಂದುವರಿ;
                }
                // Loop over all srcs of all nodes in the fusion. If the src overlaps
                // the destination and the src is not an intermediate node that's being
                // elided, then disable fusion.
                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ k = 0; k <= ctx->ಅಂಕೆ_additional_fused_ops; ++k) {
                    ಗಾಗಿ (uint32_t s = 0; s < GGML_MAX_SRC; ++s) {
                        ggml_tensor *src = cgraph->nodes[i + k]->src[s];
                        ಆದರೆ (!src || src->op == GGML_OP_NONE) {
                            ಮುಂದುವರಿ;
                        }
                        ಆದರೆ (ggml_vk_tensors_overlap(src, dst, op_srcs_fused_elementwise[k])) {
                            ತಾರ್ಕಿಕ ಸಿಕ್ಕಿತು = ಸುಳ್ಳು;
                            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ n = 0; n < k; ++n) {
                                ಆದರೆ (cgraph->nodes[i + n] == src) {
                                    ಸಿಕ್ಕಿತು = ನಿಜ;
                                    ನಿಲ್ಲಿಸು;
                                }
                            }
                            ಆದರೆ (!ಸಿಕ್ಕಿತು) {
                                need_disable = ನಿಜ;
                            }
                        }
                    }
                }
            }
            ಆದರೆ (need_disable) {
                ctx->ಅಂಕೆ_additional_fused_ops = 0;
                ctx->fused_ops_write_ಮುಖವಾಡ = 1;
                ctx->fused_topk_moe_ಬಹುಲಕ = TOPK_MOE_ಎಣಿಕೆ;
                ctx->fused_topk_moe_scale = ಸುಳ್ಳು;
                ctx->fused_topk_qsa = ಸುಳ್ಳು;
                ctx->fused_hc_post_gate = ಸುಳ್ಳು;
                ctx->fused_rms_norm_ಬಹುಲಕ = RMS_NORM_ಎಣಿಕೆ;
                fusion_string = ಶೂನ್ಯಸೂಚಕ;
            }
        }

        // Signal the almost_ready fence when the graph is mostly complete (< 20% remaining)
        ತಾರ್ಕಿಕ almost_ready = (cgraph->n_nodes - i) < cgraph->n_nodes / 5;
        ತಾರ್ಕಿಕ submit = (submitted_nodes >= ctx->device->max_nodes_ಪ್ರತಿ_submit) ||
                      (flops_ಪ್ರತಿ_submit != 0 && batch_flops >= flops_ಪ್ರತಿ_submit) ||
                      (i + ctx->ಅಂಕೆ_additional_fused_ops >= ಕೊನೆಯ_ಗಂಟು) ||
                      (almost_ready && !ctx->almost_ready_fence_pending);

        ತಾರ್ಕಿಕ enqueued = ggml_vk_build_ನಕ್ಷೆ(ctx, cgraph, i, cgraph->nodes[submit_ಗಂಟು_idx], submit_ಗಂಟು_idx, i + ctx->ಅಂಕೆ_additional_fused_ops >= ಕೊನೆಯ_ಗಂಟು, almost_ready, submit);

        ಆದರೆ (vk_perf_logger_enabled && enqueued) {
            compute_ctx = ggml_vk_get_compute_ctx(ctx);
            ಆದರೆ (!vk_perf_logger_concurrent) {
                // track a single node/fusion for the current query
                ctx->query_nodes[ctx->query_idx] = cgraph->nodes[i];
                ctx->query_fusion_ಹೆಸರುಗಳು[ctx->query_idx] = fusion_string;
                compute_ctx->s->ಬಫರ್->buf.writeTimestamp(vk::Pipeline_Stage_Flag_ಬಿಟ್ಟುಗಳು::eAllCommands, ctx->query_pool, ctx->query_idx++);
                ggml_vk_sync_buffers(ctx, compute_ctx);
            } ಇಲ್ಲದಿದ್ದರೆ {
                // track a fusion string and number of fused ops for the current node_idx
                ctx->query_fusion_ಹೆಸರುಗಳು[i] = fusion_string;
                ctx->query_fusion_ಗಂಟು_ಎಣಿಕೆ[i] = ctx->ಅಂಕೆ_additional_fused_ops;
            }
        }

        ಆದರೆ (enqueued) {
            ++submitted_nodes;

#ವ್ಯಾಖ್ಯಾನಿತವಾಗದಿದ್ದರೆ GGML_VULKAN_CHECK_RESULTS
            ಆದರೆ (ಮೊದಲ_ಗಂಟು_ಒಳ_batch) {
                ಮೊದಲ_ಗಂಟು_ಒಳ_batch = ಸುಳ್ಳು;
            }
#ಆದರೆ_ಅಂತ್ಯ
        }

        ಆದರೆ (submit && enqueued) {
            submit_after(submit_ಗಂಟು_idx, i + (ಪೂರ್ಣಾಂಕ)ctx->ಅಂಕೆ_additional_fused_ops);
        }
        i += ctx->ಅಂಕೆ_additional_fused_ops;
        ctx->ಅಂಕೆ_additional_fused_ops = 0;
        ctx->fused_ops_write_ಮುಖವಾಡ = 0;
    }

    ctx->ಕೊನೆಯ_ಒಟ್ಟು_flops = ಒಟ್ಟು_flops;

    ಆದರೆ (vk_perf_logger_enabled) {
        // End the command buffer and submit/wait
        GGML_ASSERT(!ctx->compute_ctx.expired());
        compute_ctx = ctx->compute_ctx.ಬೀಗ();
        ggml_vk_ctx_ಕೊನೆ(compute_ctx);

        ggml_vk_submit(compute_ctx, ctx->device->fence);
        VK_CHECK(ctx->device->device.waitForFences({ ctx->device->fence }, ನಿಜ, UINT64_MAX), "GGML_VULKAN_PERF waitForFences", ctx->device);
        ctx->device->device.resetFences({ ctx->device->fence });
        ctx->compute_ctx.reset();

        // Get the results and pass them to the logger
        std::vector<uint64_t> timestamps(cgraph->n_nodes + 1);
        VK_CHECK(ctx->device->device.getQueryPoolResults(ctx->query_pool, 0, ctx->query_idx, (cgraph->n_nodes + 1)*ಗಾತ್ರ(uint64_t), timestamps.ದತ್ತಾಂಶ(), ಗಾತ್ರ(uint64_t), vk::Query_ಫಲಿತಾಂಶ_Flag_ಬಿಟ್ಟುಗಳು::e64 | vk::Query_ಫಲಿತಾಂಶ_Flag_ಬಿಟ್ಟುಗಳು::eWait), "get timestamp results", ctx->device);
        ಆದರೆ (!vk_perf_logger_concurrent) {
            // Log each op separately
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 1; i < ctx->query_idx; i++) {
                ಸ್ವಯಂಚಾಲಿತ ಗಂಟು = ctx->query_nodes[i];
                ಸ್ವಯಂಚಾಲಿತ name = ctx->query_fusion_ಹೆಸರುಗಳು[i];
                ctx->perf_logger->log_timing(ಗಂಟು, name, uint64_t((timestamps[i] - timestamps[i-1]) * ctx->device->properties.limits.timestampPeriod));
            }
        } ಇಲ್ಲದಿದ್ದರೆ {
            // Log each group of nodes
            ಪೂರ್ಣಾಂಕ prev_ಗಂಟು_idx = 0;
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 1; i < ctx->query_idx; i++) {
                ಸ್ವಯಂಚಾಲಿತ cur_ಗಂಟು_idx = ctx->query_ಗಂಟು_idx[i];
                std::vector<ggml_tensor *> nodes;
                std::vector<ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *> ಹೆಸರುಗಳು;
                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ ಗಂಟು_idx = prev_ಗಂಟು_idx; ಗಂಟು_idx < cur_ಗಂಟು_idx; ++ಗಂಟು_idx) {
                    ಆದರೆ (ggml_op_is_ಖಾಲಿಗಳು(cgraph->nodes[ಗಂಟು_idx]->op)) {
                        ಮುಂದುವರಿ;
                    }
                    nodes.push_back(cgraph->nodes[ಗಂಟು_idx]);
                    ಹೆಸರುಗಳು.push_back(ctx->query_fusion_ಹೆಸರುಗಳು[ಗಂಟು_idx]);
                    ಗಂಟು_idx += ctx->query_fusion_ಗಂಟು_ಎಣಿಕೆ[ಗಂಟು_idx];
                }
                prev_ಗಂಟು_idx = cur_ಗಂಟು_idx;
                ctx->perf_logger->log_timing(nodes, ಹೆಸರುಗಳು, uint64_t((timestamps[i] - timestamps[i-1]) * ctx->device->properties.limits.timestampPeriod));
            }
        }
        ctx->perf_logger->print_timings();
    }

    ಆದರೆ (!ctx->device->support_async) {
        ggml_vk_synchronize(ctx);
    }

    ಹಿಂತಿರುಗಿ GGML_ಸ್ಥಿತಿ_SUCCESS;

    ಬಳಸದ(backend);
}

ಶೂನ್ಯ ggml_vk_ನಕ್ಷೆ_optimize(ggml_backend_t backend, ರಚನೆ ggml_cgraph * graph, ರಚನೆ ggml_backend_ನಕ್ಷೆ_optimize_params * params)
{
    VK_LOG_DEBUG("ggml_vk_graph_optimize(" << graph->n_nodes << " nodes)");
    ggml_backend_vk_context * ctx = (ggml_backend_vk_context *)backend->context;

    ಆದರೆ (ctx->device->disable_ನಕ್ಷೆ_optimize) {
        ಹಿಂತಿರುಗಿ;
    }

    ಸ್ವಯಂಚಾಲಿತ ಸ್ಥಿರಾಂಕ &is_ಖಾಲಿಗಳು = [](ಸ್ಥಿರಾಂಕ ggml_tensor * ಗಂಟು) -> ತಾರ್ಕಿಕ {
        ಹಿಂತಿರುಗಿ ಗಂಟು->op == GGML_OP_NONE || ಗಂಟು->op == GGML_OP_RESHAPE || ಗಂಟು->op == GGML_OP_TRANSPOSE || ಗಂಟು->op == GGML_OP_VIEW || ಗಂಟು->op == GGML_OP_ಕ್ರಮಬದಲಾವಣೆ;
    };

    ಸ್ವಯಂಚಾಲಿತ ಸ್ಥಿರಾಂಕ &is_src_ದ = [&is_ಖಾಲಿಗಳು](ಸ್ಥಿರಾಂಕ ggml_tensor *dst, ಸ್ಥಿರಾಂಕ ggml_tensor *src) -> ತಾರ್ಕಿಕ {
        ಸ್ವಯಂಚಾಲಿತ ಸ್ಥಿರಾಂಕ &ತಳ = [](ಸ್ಥಿರಾಂಕ ggml_tensor * tensor) {
            ಹಿಂತಿರುಗಿ tensor->view_src ? tensor->view_src : tensor;
        };
        ಗಾಗಿ (uint32_t s = 0; s < GGML_MAX_SRC; ++s) {
            ಆದರೆ (dst->src[s] == src) {
                ಹಿಂತಿರುಗಿ ನಿಜ;
            }
            ಆದರೆ (is_ಖಾಲಿಗಳು(dst) || is_ಖಾಲಿಗಳು(src)) {
                ಮುಂದುವರಿ;
            }
            // A source view of dst may read storage written through a different view by src.
            ಆದರೆ (dst->src[s] && ತಳ(dst->src[s]) == ತಳ(src)) {
                ಹಿಂತಿರುಗಿ ನಿಜ;
            }
            // Moving dst forward may overwrite storage still read through a view by src.
            ಆದರೆ (src->src[s] && ತಳ(dst) == ತಳ(src->src[s])) {
                ಹಿಂತಿರುಗಿ ನಿಜ;
            }
        }
        // implicit dependency if they view the same tensor
        ಆದರೆ (ತಳ(dst) == ತಳ(src)) {
            ಹಿಂತಿರುಗಿ ನಿಜ;
        }
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    };

    std::vector<ggml_tensor *> new_ಕ್ರಮ;
    std::vector<ತಾರ್ಕಿಕ> ಬಳಸಿದ(graph->n_nodes, ಸುಳ್ಳು);
    std::set<ggml_tensor *> ಬಳಸಿದ_ಗಂಟು_ಗಣ;

    ಪೂರ್ಣಾಂಕ ಮೊದಲ_ಬಳಸದ = 0;

    // scheduled or zero-compute nodes in [lo, hi)
    ಸ್ವಯಂಚಾಲಿತ ಸ್ಥಿರಾಂಕ &ಖಾಲಿಗಳು_or_scheduled_between = [&](ಪೂರ್ಣಾಂಕ lo, ಪೂರ್ಣಾಂಕ hi) -> ತಾರ್ಕಿಕ {
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ v = lo; v < hi; ++v) {
            ಆದರೆ (!ಬಳಸಿದ[v] && !is_ಖಾಲಿಗಳು(graph->nodes[v])) {
                ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
            }
        }
        ಹಿಂತಿರುಗಿ ನಿಜ;
    };
    ಆಗ (ಮೊದಲ_ಬಳಸದ < graph->n_nodes) {
        std::vector<ಪೂರ್ಣಾಂಕ> current_ಗಣ;

        // Check for fusion patterns and avoid reordering them
        ಸ್ವಯಂಚಾಲಿತ ಸ್ಥಿರಾಂಕ &ಹೊಂದಿಕೆ_ನಮೂನೆ = [&](ಸ್ಥಿರಾಂಕ std::initializer_list<ggml_op> &pattern, ಪೂರ್ಣಾಂಕ start) -> ತಾರ್ಕಿಕ {
            ಆದರೆ (start + (ಪೂರ್ಣಾಂಕ)pattern.size() <= graph->n_nodes) {
                ತಾರ್ಕಿಕ is_ನಮೂನೆ = ನಿಜ;
                ಗಾಗಿ (size_t j = 0; j < pattern.size(); ++j) {
                    ಆದರೆ (graph->nodes[start + j]->op != pattern.begin()[j] || ಬಳಸಿದ[start + j]) {
                        is_ನಮೂನೆ = ಸುಳ್ಳು;
                    }
                }
                ಹಿಂತಿರುಗಿ is_ನಮೂನೆ;
            }
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        };

        ಸ್ವಯಂಚಾಲಿತ ಸ್ಥಿರಾಂಕ &keep_ನಮೂನೆ = [&](ಸ್ಥಿರಾಂಕ std::initializer_list<ggml_op> &pattern) -> ತಾರ್ಕಿಕ {
            ಆದರೆ (ಹೊಂದಿಕೆ_ನಮೂನೆ(pattern, ಮೊದಲ_ಬಳಸದ)) {
                ಗಾಗಿ (size_t j = 0; j < pattern.size(); ++j) {
                    new_ಕ್ರಮ.push_back(graph->nodes[ಮೊದಲ_ಬಳಸದ + j]);
                    ಬಳಸಿದ_ಗಂಟು_ಗಣ.insert(graph->nodes[ಮೊದಲ_ಬಳಸದ + j]);
                    ಬಳಸಿದ[ಮೊದಲ_ಬಳಸದ + j] = ನಿಜ;
                }
                ಆಗ (ಮೊದಲ_ಬಳಸದ < graph->n_nodes && ಬಳಸಿದ[ಮೊದಲ_ಬಳಸದ]) {
                    ಮೊದಲ_ಬಳಸದ++;
                }
                ಹಿಂತಿರುಗಿ ನಿಜ;
            }
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        };

        ಸ್ವಯಂಚಾಲಿತ ಸ್ಥಿರಾಂಕ &ಕೂಡಿಸು_ನಮೂನೆ_alloc_deps = [&](ಸ್ಥಿರಾಂಕ std::initializer_list<ggml_op> &pattern, ಪೂರ್ಣಾಂಕ ಕೊನೆಯ_ಗಂಟು) {
            // Keep external inputs alive through the fused output.
            std::set<ggml_tensor *> ನೋಡಿದ;
            ಗಾಗಿ (size_t j = 0; j < pattern.size(); ++j) {
                ggml_tensor * ಗಂಟು = graph->nodes[ಮೊದಲ_ಬಳಸದ + j];
                ಗಾಗಿ (uint32_t s = 0; s < GGML_MAX_SRC; ++s) {
                    ggml_tensor * src = ಗಂಟು->src[s];
                    ಆದರೆ (src && ನೋಡಿದ.insert(src).second) {
                        params->ಕೂಡಿಸು_alloc_dep(params->user_ದತ್ತಾಂಶ, src, graph->nodes[ಕೊನೆಯ_ಗಂಟು]);
                    }
                }
                ನೋಡಿದ.insert(ಗಂಟು);
            }
        };

        ಸ್ವಯಂಚಾಲಿತ ಸ್ಥಿರಾಂಕ &keep_topk_moe_ನಮೂನೆ = [&](ಸ್ಥಿರಾಂಕ std::initializer_list<ggml_op> &pattern) -> ತಾರ್ಕಿಕ {
            ಆದರೆ (!ಹೊಂದಿಕೆ_ನಮೂನೆ(pattern, ಮೊದಲ_ಬಳಸದ)) {
                ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
            }

            ಪೂರ್ಣಾಂಕ ಕೊನೆಯ_ಗಂಟು = ಮೊದಲ_ಬಳಸದ + (ಪೂರ್ಣಾಂಕ) pattern.size() - 1;
            // Some TOPK_MOE variants fuse a trailing scale.
            ಆದರೆ (ಕೊನೆಯ_ಗಂಟು + 1 < graph->n_nodes && graph->nodes[ಕೊನೆಯ_ಗಂಟು + 1]->op == GGML_OP_SCALE) {
                ಕೊನೆಯ_ಗಂಟು++;
            }

            ಕೂಡಿಸು_ನಮೂನೆ_alloc_deps(pattern, ಕೊನೆಯ_ಗಂಟು);

            ಹಿಂತಿರುಗಿ keep_ನಮೂನೆ(pattern);
        };

        ಆದರೆ (keep_topk_moe_ನಮೂನೆ(topk_moe_early_softmax_norm)) {
            ಮುಂದುವರಿ;
        }
        ಆದರೆ (keep_topk_moe_ನಮೂನೆ(topk_moe_sigmoid_norm_bias)) {
            ಮುಂದುವರಿ;
        }
        ಆದರೆ (keep_topk_moe_ನಮೂನೆ(topk_moe_sqrt_softplus_norm_bias)) {
            ಮುಂದುವರಿ;
        }
        ಆದರೆ (keep_topk_moe_ನಮೂನೆ(topk_moe_early_softmax)) {
            ಮುಂದುವರಿ;
        }
        ಆದರೆ (keep_topk_moe_ನಮೂನೆ(topk_moe_late_softmax)) {
            ಮುಂದುವರಿ;
        }
        ಆದರೆ (keep_ನಮೂನೆ(snake_ನಮೂನೆ)) {
            ಮುಂದುವರಿ;
        }
        ಆದರೆ (keep_ನಮೂನೆ(topk_qsa_ನಮೂನೆ)) {
            ಮುಂದುವರಿ;
        }

        ಆದರೆ (keep_ನಮೂನೆ(rms_norm_mul_ಕೂಡಿಸು_mul_ನಮೂನೆ)) {
            ಮುಂದುವರಿ;
        }
        ಆದರೆ (keep_ನಮೂನೆ(rms_norm_mul_ಕೂಡಿಸು_ನಮೂನೆ)) {
            ಮುಂದುವರಿ;
        }
        ಆದರೆ (keep_ನಮೂನೆ(rms_norm_mul_rope_view_ಗಣ_ಅಡ್ಡಸಾಲುಗಳು_ನಮೂನೆ)) {
            ಮುಂದುವರಿ;
        }
        ಆದರೆ (keep_ನಮೂನೆ(rms_norm_view_ಗಣ_ಅಡ್ಡಸಾಲುಗಳು_ನಮೂನೆ)) {
            ಮುಂದುವರಿ;
        }
        ಆದರೆ (keep_ನಮೂನೆ(rope_view_ಗಣ_ಅಡ್ಡಸಾಲುಗಳು_ನಮೂನೆ)) {
            ಮುಂದುವರಿ;
        }
        ಆದರೆ (ಹೊಂದಿಕೆ_ನಮೂನೆ(hc_post_gate_ನಮೂನೆ, ಮೊದಲ_ಬಳಸದ)) {
            ಕೂಡಿಸು_ನಮೂನೆ_alloc_deps(hc_post_gate_ನಮೂನೆ, ಮೊದಲ_ಬಳಸದ + (ಪೂರ್ಣಾಂಕ) hc_post_gate_ನಮೂನೆ.size() - 1);
            keep_ನಮೂನೆ(hc_post_gate_ನಮೂನೆ);
            ಮುಂದುವರಿ;
        }

        // First, grab the next unused node.
        current_ಗಣ.push_back(ಮೊದಲ_ಬಳಸದ);

        // Loop through the next N nodes. Grab any that don't depend on other nodes that
        // haven't already been run. Nodes that have already been run have used[i] set
        // to true. Allow nodes that depend on the previous node if it's a fusion pattern
        // that we support (e.g. RMS_NORM + MUL).
        // This first pass only grabs "real" (non-view nodes). Second pass grabs view nodes.
        // The goal is to not interleave real and view nodes in a way that breaks fusion.
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಅಂಕೆ_ಗಮ್ಯ_CHECK = 20;
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = ಮೊದಲ_ಬಳಸದ+1; j < std::min(ಮೊದಲ_ಬಳಸದ + ಅಂಕೆ_ಗಮ್ಯ_CHECK, graph->n_nodes); ++j) {
            ಆದರೆ (ಬಳಸಿದ[j]) {
                ಮುಂದುವರಿ;
            }
            ಆದರೆ (is_ಖಾಲಿಗಳು(graph->nodes[j])) {
                ಮುಂದುವರಿ;
            }
            // Protect every interior QSA node (not just the start): the mask branch is
            // independent, so it gets pulled out and breaks keep_pattern otherwise.
            ಸ್ವಯಂಚಾಲಿತ ಸ್ಥಿರಾಂಕ &ಒಳ_qsa_ನಮೂನೆ = [&](ಪೂರ್ಣಾಂಕ n) -> ತಾರ್ಕಿಕ {
                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ o = 0; o < (ಪೂರ್ಣಾಂಕ) topk_qsa_ನಮೂನೆ.size(); ++o) {
                    ಆದರೆ (n - o >= 0 && ಹೊಂದಿಕೆ_ನಮೂನೆ(topk_qsa_ನಮೂನೆ, n - o)) {
                        ಹಿಂತಿರುಗಿ ನಿಜ;
                    }
                }
                ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
            };
            ಆದರೆ (ಹೊಂದಿಕೆ_ನಮೂನೆ(topk_moe_early_softmax_norm, j) ||
                ಹೊಂದಿಕೆ_ನಮೂನೆ(topk_moe_sigmoid_norm_bias, j) ||
                ಹೊಂದಿಕೆ_ನಮೂನೆ(topk_moe_sqrt_softplus_norm_bias, j) ||
                ಹೊಂದಿಕೆ_ನಮೂನೆ(topk_moe_early_softmax, j) ||
                ಹೊಂದಿಕೆ_ನಮೂನೆ(topk_moe_late_softmax, j) ||
                ಹೊಂದಿಕೆ_ನಮೂನೆ(snake_ನಮೂನೆ, j) ||
                ಒಳ_qsa_ನಮೂನೆ(j) ||
                ಹೊಂದಿಕೆ_ನಮೂನೆ(rms_norm_mul_ಕೂಡಿಸು_mul_ನಮೂನೆ, j) ||
                ಹೊಂದಿಕೆ_ನಮೂನೆ(rms_norm_mul_ಕೂಡಿಸು_ನಮೂನೆ, j) ||
                ಹೊಂದಿಕೆ_ನಮೂನೆ(rms_norm_mul_rope_view_ಗಣ_ಅಡ್ಡಸಾಲುಗಳು_ನಮೂನೆ, j) ||
                ಹೊಂದಿಕೆ_ನಮೂನೆ(rms_norm_view_ಗಣ_ಅಡ್ಡಸಾಲುಗಳು_ನಮೂನೆ, j) ||
                ಹೊಂದಿಕೆ_ನಮೂನೆ(rope_view_ಗಣ_ಅಡ್ಡಸಾಲುಗಳು_ನಮೂನೆ, j) ||
                ಹೊಂದಿಕೆ_ನಮೂನೆ(hc_post_gate_ನಮೂನೆ, j)) {
                ಮುಂದುವರಿ;
            }
            ತಾರ್ಕಿಕ ok = ನಿಜ;
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ c = ಮೊದಲ_ಬಳಸದ; c < j; ++c) {
                ಆದರೆ (!ಬಳಸಿದ[c] &&
                    is_src_ದ(graph->nodes[j], graph->nodes[c]) &&
                    !(c == current_ಗಣ.back() && graph->nodes[c]->op == GGML_OP_RMS_NORM && graph->nodes[j]->op == GGML_OP_MUL && ಖಾಲಿಗಳು_or_scheduled_between(c+1, j)) &&
                    !(c == current_ಗಣ.back() && graph->nodes[c]->op == GGML_OP_UNARY && graph->nodes[j]->op == GGML_OP_MUL && ಖಾಲಿಗಳು_or_scheduled_between(c+1, j)) &&
                    !(j == c+1 && c == current_ಗಣ.back() && graph->nodes[c]->op == GGML_OP_MUL_MAT && graph->nodes[j]->op == GGML_OP_ಕೂಡಿಸು) &&
                    !(j == c+1 && c == current_ಗಣ.back() && graph->nodes[c]->op == GGML_OP_MUL_MAT_ID && graph->nodes[j]->op == GGML_OP_ಕೂಡಿಸು_ID) &&
                    !(j == c+1 && c == current_ಗಣ.back() && graph->nodes[c]->op == GGML_OP_MUL_MAT_ID && graph->nodes[j]->op == GGML_OP_MUL) &&
                    !(j == c+1 && c == current_ಗಣ.back() && graph->nodes[c]->op == GGML_OP_ಕೂಡಿಸು && graph->nodes[j]->op == GGML_OP_ಕೂಡಿಸು) &&
                    !(j == c+1 && c == current_ಗಣ.back() && graph->nodes[c]->op == GGML_OP_SSM_CONV && graph->nodes[j]->op == GGML_OP_ಕೂಡಿಸು) &&
                    !(j == c+1 && c == current_ಗಣ.back() && graph->nodes[c]->op == GGML_OP_SSM_CONV && graph->nodes[j]->op == GGML_OP_UNARY)) {
                    ok = ಸುಳ್ಳು;
                    ನಿಲ್ಲಿಸು;
                }
            }
            ಆದರೆ (ok) {
                current_ಗಣ.push_back(j);

                ಪೂರ್ಣಾಂಕ rope_idx = j;

                // When we've found RMS_NORM + MUL, try to find a ROPE that uses it
                ಆದರೆ (j > 0 &&
                    graph->nodes[j]->op == GGML_OP_MUL &&
                    graph->nodes[j-1]->op == GGML_OP_RMS_NORM) {
                    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ k = j + 1; k < std::min(j + 15, graph->n_nodes); ++k) {
                        ಆದರೆ (graph->nodes[k]->op == GGML_OP_ROPE &&
                            graph->nodes[k]->src[0] == graph->nodes[j] &&
                            // Check that other srcs are already valid
                            graph->nodes[k]->src[1]->op == GGML_OP_NONE &&
                            (graph->nodes[k]->src[2] == ಶೂನ್ಯಸೂಚಕ || graph->nodes[k]->src[2]->op == GGML_OP_NONE)) {
                            rope_idx = k;
                            current_ಗಣ.push_back(rope_idx);
                            ಬಳಸಿದ[rope_idx] = ನಿಜ;
                            ನಿಲ್ಲಿಸು;
                        }
                    }
                }
                // Look for ROPE/RMS_NORM + VIEW + SET_ROWS and make them consecutive
                ಆದರೆ (graph->nodes[rope_idx]->op == GGML_OP_ROPE || graph->nodes[rope_idx]->op == GGML_OP_RMS_NORM) {
                    ಪೂರ್ಣಾಂಕ view_idx = -1;
                    ಪೂರ್ಣಾಂಕ ಗಣ_ಅಡ್ಡಸಾಲುಗಳು_idx = -1;
                    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ k = rope_idx + 1; k < std::min(rope_idx + 15, graph->n_nodes); ++k) {
                        ಆದರೆ (ಬಳಸಿದ[k]) {
                            ಮುಂದುವರಿ;
                        }
                        ಆದರೆ (view_idx == -1 && graph->nodes[k]->op == GGML_OP_VIEW && graph->nodes[k]->src[0] == graph->nodes[rope_idx]) {
                            view_idx = k;
                            ಮುಂದುವರಿ;
                        }
                        ಆದರೆ (view_idx != -1 && graph->nodes[k]->op == GGML_OP_ಗಣ_ಅಡ್ಡಸಾಲುಗಳು && graph->nodes[k]->src[0] == graph->nodes[view_idx]) {
                            ಗಣ_ಅಡ್ಡಸಾಲುಗಳು_idx = k;
                            ನಿಲ್ಲಿಸು;
                        }
                    }
                    ಆದರೆ (ಗಣ_ಅಡ್ಡಸಾಲುಗಳು_idx != -1) {
                        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಗಂಟು_idxs[] = { rope_idx, view_idx, ಗಣ_ಅಡ್ಡಸಾಲುಗಳು_idx };
                        ಸ್ಥಿರಾಂಕ ggml_op ops[] = { graph->nodes[rope_idx]->op, GGML_OP_VIEW, GGML_OP_ಗಣ_ಅಡ್ಡಸಾಲುಗಳು };
                        ತಾರ್ಕಿಕ can_pull = ggml_can_fuse_subgraph_ext(graph, ಗಂಟು_idxs, 3, ops, &ಗಣ_ಅಡ್ಡಸಾಲುಗಳು_idx, 1);

                        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ c = rope_idx + 1; can_pull && c < ಗಣ_ಅಡ್ಡಸಾಲುಗಳು_idx; ++c) {
                            ಆದರೆ (!ಬಳಸಿದ[c] && c != view_idx && !is_ಖಾಲಿಗಳು(graph->nodes[c]) &&
                                is_src_ದ(graph->nodes[ಗಣ_ಅಡ್ಡಸಾಲುಗಳು_idx], graph->nodes[c])) {
                                can_pull = ಸುಳ್ಳು;
                            }
                        }

                        ಆದರೆ (can_pull) {
                            current_ಗಣ.push_back(view_idx);
                            current_ಗಣ.push_back(ಗಣ_ಅಡ್ಡಸಾಲುಗಳು_idx);
                            ಬಳಸಿದ[view_idx] = ನಿಜ;
                            ಬಳಸಿದ[ಗಣ_ಅಡ್ಡಸಾಲುಗಳು_idx] = ನಿಜ;
                        }
                    }
                }
                // Look for MUL_MAT_ID + ADD_ID + MUL
                ಆದರೆ (j > 0 &&
                    graph->nodes[j]->op == GGML_OP_ಕೂಡಿಸು_ID &&
                    graph->nodes[j-1]->op == GGML_OP_MUL_MAT_ID) {
                    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ k = j + 1; k < std::min(j + 15, graph->n_nodes); ++k) {
                        ಆದರೆ (graph->nodes[k]->op == GGML_OP_MUL &&
                            graph->nodes[k]->src[0] == graph->nodes[j] &&
                            // src1 must either be weights or already processed
                            (graph->nodes[k]->src[1]->op == GGML_OP_NONE || ಬಳಸಿದ_ಗಂಟು_ಗಣ.find(graph->nodes[k]->src[1]) != ಬಳಸಿದ_ಗಂಟು_ಗಣ.end())) {
                            current_ಗಣ.push_back(k);
                            ಬಳಸಿದ[k] = ನಿಜ;
                            ನಿಲ್ಲಿಸು;
                        }
                    }
                }
                // Look for MUL_MAT + ADD + ADD
                ಆದರೆ (j > 0 &&
                    graph->nodes[j]->op == GGML_OP_ಕೂಡಿಸು &&
                    graph->nodes[j-1]->op == GGML_OP_MUL_MAT) {
                    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ k = j + 1; k < std::min(j + 15, graph->n_nodes); ++k) {
                        ಆದರೆ (graph->nodes[k]->op == GGML_OP_ಕೂಡಿಸು &&
                            graph->nodes[k]->src[0] == graph->nodes[j] &&
                            // src1 must either be weights or already processed
                            (graph->nodes[k]->src[1]->op == GGML_OP_NONE || ಬಳಸಿದ_ಗಂಟು_ಗಣ.find(graph->nodes[k]->src[1]) != ಬಳಸಿದ_ಗಂಟು_ಗಣ.end())) {
                            current_ಗಣ.push_back(k);
                            ಬಳಸಿದ[k] = ನಿಜ;
                            ನಿಲ್ಲಿಸು;
                        }
                    }
                }
                // SSM_CONV + ADD + UNARY: pull the consuming UNARY forward
                ಆದರೆ (j > 0 &&
                    graph->nodes[j]->op == GGML_OP_ಕೂಡಿಸು &&
                    graph->nodes[j-1]->op == GGML_OP_SSM_CONV) {
                    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ k = j + 1; k < std::min(j + 15, graph->n_nodes); ++k) {
                        ಆದರೆ (graph->nodes[k]->op == GGML_OP_UNARY &&
                            graph->nodes[k]->src[0] == graph->nodes[j]) {
                            current_ಗಣ.push_back(k);
                            ಬಳಸಿದ[k] = ನಿಜ;
                            ನಿಲ್ಲಿಸು;
                        }
                    }
                }
                // UNARY + MUL: pull the consuming MUL forward
                ಆದರೆ (j > 0 &&
                    graph->nodes[j]->op == GGML_OP_UNARY) {
                    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ k = j + 1; k < std::min(j + 15, graph->n_nodes); ++k) {
                        ggml_tensor * mul = graph->nodes[k];
                        ಆದರೆ (mul->op != GGML_OP_MUL || (mul->src[0] != graph->nodes[j] && mul->src[1] != graph->nodes[j])) {
                            ಮುಂದುವರಿ;
                        }
                        ggml_tensor * other = (mul->src[0] == graph->nodes[j]) ? mul->src[1] : mul->src[0];
                        // the other src must either be weights or already processed
                        ಆದರೆ (!(other->op == GGML_OP_NONE || ಬಳಸಿದ_ಗಂಟು_ಗಣ.find(other) != ಬಳಸಿದ_ಗಂಟು_ಗಣ.end())) {
                            ಮುಂದುವರಿ;
                        }
                        ಆದರೆ (!ggml_vk_can_fuse_unary_mul(graph, j, k)) {
                            ಮುಂದುವರಿ;
                        }
                        current_ಗಣ.push_back(k);
                        ಬಳಸಿದ[k] = ನಿಜ;
                        ನಿಲ್ಲಿಸು;
                    }
                }
            }
        }
        // Second pass grabs view nodes.
        // Skip this if it would break a fusion optimization (don't split up add->rms_norm or add->add).
        ಆದರೆ (graph->nodes[current_ಗಣ.back()]->op != GGML_OP_ಕೂಡಿಸು) {
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = ಮೊದಲ_ಬಳಸದ+1; j < std::min(ಮೊದಲ_ಬಳಸದ + ಅಂಕೆ_ಗಮ್ಯ_CHECK, graph->n_nodes); ++j) {
                ಆದರೆ (ಬಳಸಿದ[j]) {
                    ಮುಂದುವರಿ;
                }
                ಆದರೆ (!is_ಖಾಲಿಗಳು(graph->nodes[j])) {
                    ಮುಂದುವರಿ;
                }
                ತಾರ್ಕಿಕ ok = ನಿಜ;
                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ c = ಮೊದಲ_ಬಳಸದ; c < j; ++c) {
                    ತಾರ್ಕಿಕ c_ಒಳ_current_ಗಣ = std::find(current_ಗಣ.begin(), current_ಗಣ.end(), c) != current_ಗಣ.end();
                    // skip views whose srcs haven't been processed.
                    ಆದರೆ (!ಬಳಸಿದ[c] &&
                        is_src_ದ(graph->nodes[j], graph->nodes[c]) &&
                        !c_ಒಳ_current_ಗಣ) {
                        ok = ಸುಳ್ಳು;
                        ನಿಲ್ಲಿಸು;
                    }
                }
                ಆದರೆ (ok) {
                    current_ಗಣ.push_back(j);
                }
            }
        }

        // Push the current set into new_order
        ಗಾಗಿ (ಸ್ವಯಂಚಾಲಿತ c : current_ಗಣ) {
            new_ಕ್ರಮ.push_back(graph->nodes[c]);
            ಬಳಸಿದ_ಗಂಟು_ಗಣ.insert(graph->nodes[c]);
            ಬಳಸಿದ[c] = ನಿಜ;
        }
        ಆಗ (ಮೊದಲ_ಬಳಸದ < graph->n_nodes && ಬಳಸಿದ[ಮೊದಲ_ಬಳಸದ]) {
            ಮೊದಲ_ಬಳಸದ++;
        }
    }
    // Replace the graph with the new order.
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < graph->n_nodes; ++i) {
        graph->nodes[i] = new_ಕ್ರಮ[i];
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_vk_event_ದಾಖಲೆ(ggml_backend_t backend, ggml_backend_event_t event) {
    VK_LOG_DEBUG("ggml_backend_vk_event_record(backend=" << backend << ", event=" << event << ")");
    ggml_backend_vk_context * ctx = (ggml_backend_vk_context *)backend->context;
    vk_event *vkev = (vk_event *)event->context;

    ggml_vk_submit_transfer_ctx(ctx);

    vk_context compute_ctx = ggml_vk_get_compute_ctx(ctx);
    ಸ್ವಯಂಚಾಲಿತ* cmd_buf = compute_ctx->s->ಬಫರ್; // retrieve pointer before it gets reset

    ಆದರೆ (vkev->has_event) {
        // Move existing event into submitted
        vkev->events_submitted.push_back(vkev->event);
    }

    // Grab the next event and record it, create one if necessary
    ಆದರೆ (vkev->events_free.ಖಾಲಿಗಳು()) {
        vkev->event = ctx->device->device.createEvent({});
    } ಇಲ್ಲದಿದ್ದರೆ {
        vkev->event = vkev->events_free.back();
        vkev->events_free.pop_back();
    }

    vkev->has_event = ನಿಜ;

    ggml_vk_ಗಣ_event(compute_ctx, vkev->event);

    vkev->tl_semaphore.value++;
    compute_ctx->s->signal_semaphores.push_back(vkev->tl_semaphore);
    ggml_vk_ctx_ಕೊನೆ(compute_ctx);

    ggml_vk_submit(compute_ctx, {});
    ctx->submit_pending = ನಿಜ;
    vkev->cmd_ಬಫರ್ = cmd_buf;
    vkev->cmd_ಬಫರ್_use_counter = cmd_buf->use_counter;
    ctx->compute_ctx.reset();
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_vk_event_wait(ggml_backend_t backend, ggml_backend_event_t event) {
    VK_LOG_DEBUG("ggml_backend_vk_event_wait(backend=" << backend << ", event=" << event << ")");
    ggml_backend_vk_context * ctx = (ggml_backend_vk_context *)backend->context;
    vk_event *vkev = (vk_event *)event->context;

    vk_context compute_ctx = ggml_vk_get_compute_ctx(ctx);

    ಆದರೆ (vkev->has_event) {
        // Wait for latest event
        ggml_vk_wait_events(compute_ctx, { vkev->event });

        ಆದರೆ (ctx->device->async_use_transfer_ಸರತಿ) {
            vk_context transfer_ctx = ggml_vk_get_transfer_ctx(ctx);
            transfer_ctx->s->wait_semaphores.push_back(vkev->tl_semaphore);
        }
    }
}

ಸ್ಥಿರ ggml_backend_i ggml_backend_vk_interface = {
    /* .get_name                = */ ggml_backend_vk_name,
    /* .free                    = */ ggml_backend_vk_free,
    /* .set_tensor_async        = */ ggml_backend_vk_ಗಣ_tensor_async,
    /* .get_tensor_async        = */ ggml_backend_vk_get_tensor_async,
    /* .set_tensor_2d_async     = */ ggml_backend_vk_ಗಣ_tensor_2d_async,
    /* .get_tensor_2d_async     = */ ggml_backend_vk_get_tensor_2d_async,
    /* .cpy_tensor_async        = */ ggml_backend_vk_cpy_tensor_async,
    /* .synchronize             = */ ggml_backend_vk_synchronize,
    /* .graph_plan_create       = */ ಖಾಲಿ,
    /* .graph_plan_free         = */ ಖಾಲಿ,
    /* .graph_plan_update       = */ ಖಾಲಿ,
    /* .graph_plan_compute      = */ ಖಾಲಿ,
    /* .graph_compute           = */ ggml_backend_vk_ನಕ್ಷೆ_compute,
    /* .event_record            = */ ggml_backend_vk_event_ದಾಖಲೆ,
    /* .event_wait              = */ ggml_backend_vk_event_wait,
    /* .graph_optimize          = */ ggml_vk_ನಕ್ಷೆ_optimize,
};

ಸ್ಥಿರ ggml_guid_t ggml_backend_vk_guid() {
    ಸ್ಥಿರ ggml_guid guid = { 0xb8, 0xf7, 0x4f, 0x86, 0x40, 0x3c, 0xe1, 0x02, 0x91, 0xc8, 0xdd, 0xe9, 0x02, 0x3f, 0xc0, 0x2b };
    ಹಿಂತಿರುಗಿ &guid;
}

ggml_backend_t ggml_backend_vk_init(size_t dev_ಅಂಕೆ) {
    VK_LOG_DEBUG("ggml_backend_vk_init(" << dev_ಅಂಕೆ << ")");

    ggml_backend_vk_context * ctx = ಹೊಸ ggml_backend_vk_context;
    ggml_vk_init(ctx, dev_ಅಂಕೆ);

    ggml_backend_t vk_backend = ಹೊಸ ggml_backend {
        /* .guid    = */ ggml_backend_vk_guid(),
        /* .iface   = */ ggml_backend_vk_interface,
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_vk_reg(), dev_ಅಂಕೆ),
        /* .context = */ ctx,
    };

    ಆದರೆ (!ctx->device->support_async) {
        vk_backend->iface.get_tensor_async = ಶೂನ್ಯಸೂಚಕ;
    }

    ಹಿಂತಿರುಗಿ vk_backend;
}

ತಾರ್ಕಿಕ ggml_backend_is_vk(ggml_backend_t backend) {
    ಹಿಂತಿರುಗಿ backend != ಖಾಲಿ && ggml_guid_matches(backend->guid, ggml_backend_vk_guid());
}

ಪೂರ್ಣಾಂಕ ggml_backend_vk_get_device_ಎಣಿಕೆ() {
    ಹಿಂತಿರುಗಿ ggml_vk_get_device_ಎಣಿಕೆ();
}

ಶೂನ್ಯ ggml_backend_vk_get_device_description(ಪೂರ್ಣಾಂಕ device, ಅಕ್ಷರ * description, size_t description_ಗಾತ್ರ) {
    GGML_ASSERT(device < (ಪೂರ್ಣಾಂಕ) vk_instance.device_indices.size());
    ಪೂರ್ಣಾಂಕ dev_idx = vk_instance.device_indices[device];
    ggml_vk_get_device_description(dev_idx, description, description_ಗಾತ್ರ);
}

ಶೂನ್ಯ ggml_backend_vk_get_device_ಸ್ಮರಣೆ(ಪೂರ್ಣಾಂಕ device, size_t * ಬಿಡುಗಡೆ, size_t * ಒಟ್ಟು) {
    GGML_ASSERT(device < (ಪೂರ್ಣಾಂಕ) vk_instance.device_indices.size());
    GGML_ASSERT(device < (ಪೂರ್ಣಾಂಕ) vk_instance.device_supports_membudget.size());

    vk::PhysicalDevice vkdev = vk_instance.instance.enumeratePhysicalDevices()[vk_instance.device_indices[device]];
    vk::Physical_Device_ಸ್ಮರಣೆ_Budget_Properties_EXT budgetprops;
    vk::Physical_Device_ಸ್ಮರಣೆ_Properties2 memprops = {};
    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ membudget_supported = vk_instance.device_supports_membudget[device];
    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ is_integrated_gpu = vkdev.getProperties().deviceType == vk::PhysicalDeviceType::eIntegratedGpu;

    ಆದರೆ (membudget_supported) {
        memprops.p_ಮುಂದಿನ = &budgetprops;
    }
    vkdev.get_ಸ್ಮರಣೆ_Properties2(&memprops);

    *ಒಟ್ಟು = 0;
    *ಬಿಡುಗಡೆ = 0;

    ಗಾಗಿ (uint32_t i = 0; i < memprops.ಸ್ಮರಣೆ_Properties.ಸ್ಮರಣೆ_ರಾಶಿ_ಎಣಿಕೆ; ++i) {
        ಸ್ಥಿರಾಂಕ vk::ಸ್ಮರಣೆ_ರಾಶಿ & ರಾಶಿ_2 = memprops.ಸ್ಮರಣೆ_Properties.ಸ್ಮರಣೆ_Heaps[i];

        ಆದರೆ (is_integrated_gpu || (ರಾಶಿ_2.flags & vk::ಸ್ಮರಣೆ_ರಾಶಿ_Flag_ಬಿಟ್ಟುಗಳು::eDeviceLocal)) {
            *ಒಟ್ಟು += ರಾಶಿ_2.size;

            ಆದರೆ (membudget_supported && i < budgetprops.ರಾಶಿ_Usage.size()) {
                // Driver may report heapBudget > heap.size so we add a safeguard here
                // See https://github.com/ggml-org/llama.cpp/issues/29277
                ಸ್ಥಿರಾಂಕ vk::Device_ಗಾತ್ರ budget = std::min(budgetprops.ರಾಶಿ_Budget[i], ರಾಶಿ_2.size);
                *ಬಿಡುಗಡೆ += budget > budgetprops.ರಾಶಿ_Usage[i] ? budget - budgetprops.ರಾಶಿ_Usage[i] : 0;
            } ಇಲ್ಲದಿದ್ದರೆ {
                *ಬಿಡುಗಡೆ += ರಾಶಿ_2.size;
            }
        }
    }
}

ಸ್ಥಿರ vk::PhysicalDeviceType ggml_backend_vk_get_device_type(ಪೂರ್ಣಾಂಕ device_idx) {
    GGML_ASSERT(device_idx >= 0 && device_idx < (ಪೂರ್ಣಾಂಕ) vk_instance.device_indices.size());

    vk::PhysicalDevice device = vk_instance.instance.enumeratePhysicalDevices()[vk_instance.device_indices[device_idx]];

    vk::PhysicalDeviceProperties2 props = {};
    device.getProperties2(&props);

    ಹಿಂತಿರುಗಿ props.properties.deviceType;
}

ಸ್ಥಿರ std::string ggml_backend_vk_get_device_pci_id(ಪೂರ್ಣಾಂಕ device_idx) {
    GGML_ASSERT(device_idx >= 0 && device_idx < (ಪೂರ್ಣಾಂಕ) vk_instance.device_indices.size());

    vk::PhysicalDevice device = vk_instance.instance.enumeratePhysicalDevices()[vk_instance.device_indices[device_idx]];

    ಸ್ಥಿರಾಂಕ std::vector<vk::ExtensionProperties> ext_props = device.enumerateDeviceExtensionProperties();

    ತಾರ್ಕಿಕ ext_support = ಸುಳ್ಳು;

    ಗಾಗಿ (ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ& properties : ext_props) {
        ಆದರೆ (strcmp("VK_EXT_pci_bus_info", properties.extensionName) == 0) {
            ext_support = ನಿಜ;
            ನಿಲ್ಲಿಸು;
        }
    }

    ಆದರೆ (!ext_support) {
        ಹಿಂತಿರುಗಿ "";
    }

    vk::PhysicalDeviceProperties2 props = {};
    vk::PhysicalDevicePCIBusInfoPropertiesEXT pci_bus_info = {};

    props.p_ಮುಂದಿನ = &pci_bus_info;

    device.getProperties2(&props);

    ಸ್ಥಿರಾಂಕ uint32_t pci_domain = pci_bus_info.pciDomain;
    ಸ್ಥಿರಾಂಕ uint32_t pci_bus = pci_bus_info.pciBus;
    ಸ್ಥಿರಾಂಕ uint32_t pci_device = pci_bus_info.pciDevice;
    ಸ್ಥಿರಾಂಕ uint8_t pci_function = (uint8_t) pci_bus_info.pciFunction; // pci function is between 0 and 7, prevent printf overflow warning

    ಅಕ್ಷರ pci_bus_id[16] = {};
    snprintf(pci_bus_id, ಗಾತ್ರ(pci_bus_id), "%04x:%02x:%02x.%x", pci_domain, pci_bus, pci_device, pci_function);

    ಹಿಂತಿರುಗಿ std::string(pci_bus_id);
}

ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * ggml_backend_vk_device_get_name(ggml_backend_dev_t dev) {
    ggml_backend_vk_device_context * ctx = (ggml_backend_vk_device_context *)dev->context;
    ಹಿಂತಿರುಗಿ ctx->name.c_str();
}

ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * ggml_backend_vk_device_get_description(ggml_backend_dev_t dev) {
    ggml_backend_vk_device_context * ctx = (ggml_backend_vk_device_context *)dev->context;
    ಹಿಂತಿರುಗಿ ctx->description.c_str();
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_vk_device_get_ಸ್ಮರಣೆ(ggml_backend_dev_t device, size_t * ಬಿಡುಗಡೆ, size_t * ಒಟ್ಟು) {
    ggml_backend_vk_device_context * ctx = (ggml_backend_vk_device_context *)device->context;
    ggml_backend_vk_get_device_ಸ್ಮರಣೆ(ctx->device, ಬಿಡುಗಡೆ, ಒಟ್ಟು);
}

ಸ್ಥಿರ ggml_backend_ಬಫರ್_type_t ggml_backend_vk_device_get_ಬಫರ್_type(ggml_backend_dev_t dev) {
    ggml_backend_vk_device_context * ctx = (ggml_backend_vk_device_context *)dev->context;
    ಹಿಂತಿರುಗಿ ggml_backend_vk_ಬಫರ್_type(ctx->device);
}

ಸ್ಥಿರ ggml_backend_ಬಫರ್_type_t ggml_backend_vk_device_get_host_ಬಫರ್_type(ggml_backend_dev_t dev) {
    ಬಳಸದ(dev);
    ಹಿಂತಿರುಗಿ ggml_backend_vk_host_ಬಫರ್_type();
}

ಸ್ಥಿರ ಎಣಿಕೆ ggml_backend_dev_type ggml_backend_vk_device_get_type(ggml_backend_dev_t dev) {
    ggml_backend_vk_device_context * ctx = (ggml_backend_vk_device_context *)dev->context;

    ಹಿಂತಿರುಗಿ ctx->is_integrated_gpu ? GGML_BACKEND_DEVICE_TYPE_IGPU : GGML_BACKEND_DEVICE_TYPE_GPU;
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_vk_device_get_props(ggml_backend_dev_t dev, ರಚನೆ ggml_backend_dev_props * props) {
    ggml_backend_vk_device_context * ctx = (ggml_backend_vk_device_context *)dev->context;

    props->name        = ggml_backend_vk_device_get_name(dev);
    props->description = ggml_backend_vk_device_get_description(dev);
    props->type        = ggml_backend_vk_device_get_type(dev);
    props->device_id   = ctx->pci_bus_id.ಖಾಲಿಗಳು() ? ಶೂನ್ಯಸೂಚಕ : ctx->pci_bus_id.c_str();
    ggml_backend_vk_device_get_ಸ್ಮರಣೆ(dev, &props->ಸ್ಮರಣೆ_free, &props->ಸ್ಮರಣೆ_ಒಟ್ಟು);
    props->caps = {
        /* .async                 = */ ನಿಜ,
        /* .host_buffer           = */ ನಿಜ,
        /* .buffer_from_host_ptr  = */ ಸುಳ್ಳು,
        /* .events                = */ ನಿಜ,
        /* .mmap_support          = */ !ctx->is_integrated_gpu,
    };
}

ಸ್ಥಿರ ggml_backend_t ggml_backend_vk_device_init(ggml_backend_dev_t dev, ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * params) {
    ಬಳಸದ(params);
    ggml_backend_vk_device_context * ctx = (ggml_backend_vk_device_context *)dev->context;
    ಹಿಂತಿರುಗಿ ggml_backend_vk_init(ctx->device);
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_backend_vk_device_supports_op(ggml_backend_dev_t dev, ಸ್ಥಿರಾಂಕ ggml_tensor * op) {
    ggml_backend_vk_device_context * ctx = (ggml_backend_vk_device_context *)dev->context;
    ಸ್ಥಿರಾಂಕ vk_device& device = ggml_vk_get_device(ctx->device);

    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ ಬಳಸುತ್ತದೆ_bda = (op->op == GGML_OP_IM2_ಲಂಬಸಾಲು || op->op == GGML_OP_IM2_ಲಂಬಸಾಲು_3_D) &&
                          device->shader_int64 && device->ಬಫರ್_device_ವಿಳಾಸ;

    ಸ್ವಯಂಚಾಲಿತ ಸ್ಥಿರಾಂಕ & tensor_ಗಾತ್ರ_supported = [&](size_t tensor_ಗಾತ್ರ) {
        ಆದರೆ (tensor_ಗಾತ್ರ > device->max_ಬಫರ್_ಗಾತ್ರ) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }
        // For im2col shaders using BDA, maxStorageBufferRange limit doesn't apply.
        // If shader64BitIndexing is enabled, maxStorageBufferRange limit doesn't apply.
        ಆದರೆ (!ಬಳಸುತ್ತದೆ_bda && !device->shader_64b_indexing) {
            ಆದರೆ (tensor_ಗಾತ್ರ > device->properties.limits.max_Storage_ಬಫರ್_Range) {
                ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
            }
        }
        ಹಿಂತಿರುಗಿ ನಿಜ;
    };
    // reject any tensors larger than the max buffer size
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < GGML_MAX_SRC; i++) {
        ಆದರೆ (op->src[i] && !tensor_ಗಾತ್ರ_supported(ggml_nbytes(op->src[i]))) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }
    }
    ಆದರೆ (!tensor_ಗಾತ್ರ_supported(ggml_nbytes(op))) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಆಯ್ಕೆ (op->op) {
        ಸ್ಥಿತಿ GGML_OP_UNARY:
            ಆಯ್ಕೆ (ggml_get_unary_op(op)) {
                ಸ್ಥಿತಿ GGML_UNARY_OP_EXP:
                ಸ್ಥಿತಿ GGML_UNARY_OP_EXPM1:
                ಸ್ಥಿತಿ GGML_UNARY_OP_ELU:
                ಸ್ಥಿತಿ GGML_UNARY_OP_GELU:
                ಸ್ಥಿತಿ GGML_UNARY_OP_GELU_ERF:
                ಸ್ಥಿತಿ GGML_UNARY_OP_GELU_ಶೀಘ್ರ:
                ಸ್ಥಿತಿ GGML_UNARY_OP_SILU:
                ಸ್ಥಿತಿ GGML_UNARY_OP_RELU:
                ಸ್ಥಿತಿ GGML_UNARY_OP_XIELU:
                ಸ್ಥಿತಿ GGML_UNARY_OP_NEG:
                ಸ್ಥಿತಿ GGML_UNARY_OP_TANH:
                ಸ್ಥಿತಿ GGML_UNARY_OP_SIGMOID:
                ಸ್ಥಿತಿ GGML_UNARY_OP_HARDSIGMOID:
                ಸ್ಥಿತಿ GGML_UNARY_OP_HARDSWISH:
                ಸ್ಥಿತಿ GGML_UNARY_OP_ABS:
                ಸ್ಥಿತಿ GGML_UNARY_OP_SOFTPLUS:
                ಸ್ಥಿತಿ GGML_UNARY_OP_ಹೆಜ್ಜೆ:
                ಸ್ಥಿತಿ GGML_UNARY_OP_ಸುತ್ತು:
                ಸ್ಥಿತಿ GGML_UNARY_OP_CEIL:
                ಸ್ಥಿತಿ GGML_UNARY_OP_FLOOR:
                ಸ್ಥಿತಿ GGML_UNARY_OP_TRUNC:
                ಸ್ಥಿತಿ GGML_UNARY_OP_SGN:
                    ಹಿಂತಿರುಗಿ (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16) &&
                           (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) &&
                           (op->src[0]->type == op->type);
                ಪೂರ್ವನಿಯೋಜಿತ:
                    ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
            }
        ಸ್ಥಿತಿ GGML_OP_GLU:
            ಆಯ್ಕೆ (ggml_get_glu_op(op)) {
                ಸ್ಥಿತಿ GGML_GLU_OP_GEGLU:
                ಸ್ಥಿತಿ GGML_GLU_OP_REGLU:
                ಸ್ಥಿತಿ GGML_GLU_OP_SWIGLU:
                ಸ್ಥಿತಿ GGML_GLU_OP_SWIGLU_OAI:
                ಸ್ಥಿತಿ GGML_GLU_OP_GEGLU_ERF:
                ಸ್ಥಿತಿ GGML_GLU_OP_GEGLU_ಶೀಘ್ರ:
                ಸ್ಥಿತಿ GGML_GLU_OP_SWIGLU_CLAMP:
                    ಹಿಂತಿರುಗಿ (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16) &&
                           (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) &&
                           (op->src[0]->type == op->type) &&
                           (!op->src[1] || op->src[1]->type == op->src[0]->type);
                ಪೂರ್ವನಿಯೋಜಿತ:
                    ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
            }
        ಸ್ಥಿತಿ GGML_OP_MUL_MAT:
        ಸ್ಥಿತಿ GGML_OP_MUL_MAT_ID:
            {
                ggml_type src0_type = op->src[0]->type;
                ಆದರೆ (op->op == GGML_OP_MUL_MAT_ID) {
                    ಆದರೆ (!ggml_vk_use_mul_mat_vec_id(op)) {
                        ಸ್ಥಿರಾಂಕ ggml_tensor * ids = op->src[2];
                        // Shaders store row IDs as uint16_t slot and token indices.
                        ಆದರೆ (ids->ne[0] > int64_t(UINT16_MAX) + 1 || ids->ne[1] > int64_t(UINT16_MAX) + 1) {
                            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
                        }
                    }
                    ಆದರೆ (!device->mul_mat_id_s[src0_type] && !device->mul_mat_id_m[src0_type] && !device->mul_mat_id_l[src0_type]) {
                        // If there's not enough shared memory for row_ids and the result tile, fallback to CPU
                        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
                    }
                    ಆದರೆ (ggml_get_op_params_i32(op, 3) == GGML_PREC_F32) {
                        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
                    }
                }
                ಆಯ್ಕೆ (src0_type) {
                    ಸ್ಥಿತಿ GGML_TYPE_F32:
                    ಸ್ಥಿತಿ GGML_TYPE_F16:
                    ಸ್ಥಿತಿ GGML_TYPE_BF16:
                    ಸ್ಥಿತಿ GGML_TYPE_Q1_0:
                    ಸ್ಥಿತಿ GGML_TYPE_Q2_0:
                    ಸ್ಥಿತಿ GGML_TYPE_Q4_0:
                    ಸ್ಥಿತಿ GGML_TYPE_Q4_1:
                    ಸ್ಥಿತಿ GGML_TYPE_Q5_0:
                    ಸ್ಥಿತಿ GGML_TYPE_Q5_1:
                    ಸ್ಥಿತಿ GGML_TYPE_Q8_0:
                    ಸ್ಥಿತಿ GGML_TYPE_Q2_K:
                    ಸ್ಥಿತಿ GGML_TYPE_Q3_K:
                    ಸ್ಥಿತಿ GGML_TYPE_Q4_K:
                    ಸ್ಥಿತಿ GGML_TYPE_Q5_K:
                    ಸ್ಥಿತಿ GGML_TYPE_Q6_K:
                    ಸ್ಥಿತಿ GGML_TYPE_IQ1_S:
                    ಸ್ಥಿತಿ GGML_TYPE_IQ1_M:
                    ಸ್ಥಿತಿ GGML_TYPE_IQ2_XXS:
                    ಸ್ಥಿತಿ GGML_TYPE_IQ2_XS:
                    ಸ್ಥಿತಿ GGML_TYPE_IQ2_S:
                    ಸ್ಥಿತಿ GGML_TYPE_IQ3_XXS:
                    ಸ್ಥಿತಿ GGML_TYPE_IQ3_S:
                    ಸ್ಥಿತಿ GGML_TYPE_IQ4_XS:
                    ಸ್ಥಿತಿ GGML_TYPE_IQ4_NL:
                    ಸ್ಥಿತಿ GGML_TYPE_MXFP4:
                    ಸ್ಥಿತಿ GGML_TYPE_NVFP4:
                    ಸ್ಥಿತಿ GGML_TYPE_TQ1_0:
                    ಸ್ಥಿತಿ GGML_TYPE_TQ2_0:
                        ನಿಲ್ಲಿಸು;
                    ಪೂರ್ವನಿಯೋಜಿತ:
                        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
                }
                ರಚನೆ ggml_tensor * a;
                ರಚನೆ ggml_tensor * b;
                ಆದರೆ (op->op == GGML_OP_MUL_MAT) {
                    a = op->src[0];
                    b = op->src[1];
                } ಇಲ್ಲದಿದ್ದರೆ {
                    a = op->src[2];
                    b = op->src[1];
                }
                ಆದರೆ (a->ne[3] != b->ne[3]) {
                    ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
                }
                ಆದರೆ (!(ggml_vk_dim01_contiguous(op->src[0]) || op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16 || op->src[0]->type == GGML_TYPE_BF16) ||
                    !(ggml_vk_dim01_contiguous(op->src[1]) || op->src[1]->type == GGML_TYPE_F32 || op->src[1]->type == GGML_TYPE_F16)) {
                    ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
                }
                ಆದರೆ (op->src[0]->type == GGML_TYPE_BF16 && op->src[1]->type == GGML_TYPE_F16) {
                    // We currently don't have a bf16 x f16 shader, or an fp16->bf16 copy shader.
                    // So don't support this combination for now.
                    ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
                }
                ಆದರೆ (op->src[1]->type == GGML_TYPE_BF16 && op->src[0]->type != GGML_TYPE_BF16 && op->op != GGML_OP_MUL_MAT) {
                    // BF16 in src1 (widened to F32/F16) is only wired up for MUL_MAT
                    ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
                }

                ಹಿಂತಿರುಗಿ ನಿಜ;
            }
        ಸ್ಥಿತಿ GGML_OP_FLASH_ATTN_EXT:
            {
                ತಾರ್ಕಿಕ coopmat2 = device->coopmat2;
                uint32_t HSK = op->src[1]->ne[0];
                uint32_t HSV = op->src[2]->ne[0];
                ಆದರೆ ((HSK % 8) != 0 || (HSV % 8) != 0) {
                    ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
                }
                ಆದರೆ (op->src[4] && op->src[4]->type != GGML_TYPE_F32) {
                    ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
                }
                ಆದರೆ (op->src[0]->type != GGML_TYPE_F32) {
                    ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
                }
                ಆದರೆ (op->type != GGML_TYPE_F32) {
                    ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
                }
                ಆದರೆ (op->src[3] && op->src[3]->type != GGML_TYPE_F16) {
                    ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
                }
                ಸ್ವಯಂಚಾಲಿತ fa_kv_ok = [](ggml_type t) {
                    ಆಯ್ಕೆ (t) {
                    ಸ್ಥಿತಿ GGML_TYPE_F32:
                    ಸ್ಥಿತಿ GGML_TYPE_F16:
                    ಸ್ಥಿತಿ GGML_TYPE_BF16:
                    ಸ್ಥಿತಿ GGML_TYPE_Q8_0:
                    ಸ್ಥಿತಿ GGML_TYPE_Q5_1:
                    ಸ್ಥಿತಿ GGML_TYPE_Q5_0:
                    ಸ್ಥಿತಿ GGML_TYPE_Q4_1:
                    ಸ್ಥಿತಿ GGML_TYPE_Q4_0:
                    ಸ್ಥಿತಿ GGML_TYPE_IQ4_NL:
                        ಹಿಂತಿರುಗಿ ನಿಜ;
                    ಪೂರ್ವನಿಯೋಜಿತ:
                        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
                    }
                };
                ಆದರೆ (!fa_kv_ok(op->src[1]->type) || !fa_kv_ok(op->src[2]->type)) {
                    ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
                }
                ಆದರೆ ((op->src[1]->type == GGML_TYPE_BF16) != (op->src[2]->type == GGML_TYPE_BF16)) {
                    ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
                }
                ಆದರೆ (!coopmat2 && !(device->subgroup_shuffle && device->subgroup_vote)) {
                    // scalar/coopmat1 FA uses subgroupShuffle/subgroupAll
                    ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
                }
                ಹಿಂತಿರುಗಿ ನಿಜ;
            }
        ಸ್ಥಿತಿ GGML_OP_GET_ಅಡ್ಡಸಾಲುಗಳು:
            {
                ಆಯ್ಕೆ (op->src[0]->type) {
                    ಸ್ಥಿತಿ GGML_TYPE_F32:
                    ಸ್ಥಿತಿ GGML_TYPE_F16:
                    ಸ್ಥಿತಿ GGML_TYPE_BF16:
                    ಸ್ಥಿತಿ GGML_TYPE_Q1_0:
                    ಸ್ಥಿತಿ GGML_TYPE_Q2_0:
                    ಸ್ಥಿತಿ GGML_TYPE_Q4_0:
                    ಸ್ಥಿತಿ GGML_TYPE_Q4_1:
                    ಸ್ಥಿತಿ GGML_TYPE_Q5_0:
                    ಸ್ಥಿತಿ GGML_TYPE_Q5_1:
                    ಸ್ಥಿತಿ GGML_TYPE_Q8_0:
                    ಸ್ಥಿತಿ GGML_TYPE_Q2_K:
                    ಸ್ಥಿತಿ GGML_TYPE_Q3_K:
                    ಸ್ಥಿತಿ GGML_TYPE_Q4_K:
                    ಸ್ಥಿತಿ GGML_TYPE_Q5_K:
                    ಸ್ಥಿತಿ GGML_TYPE_Q6_K:
                    ಸ್ಥಿತಿ GGML_TYPE_IQ1_S:
                    ಸ್ಥಿತಿ GGML_TYPE_IQ1_M:
                    ಸ್ಥಿತಿ GGML_TYPE_IQ2_XXS:
                    ಸ್ಥಿತಿ GGML_TYPE_IQ2_XS:
                    ಸ್ಥಿತಿ GGML_TYPE_IQ2_S:
                    ಸ್ಥಿತಿ GGML_TYPE_IQ3_XXS:
                    ಸ್ಥಿತಿ GGML_TYPE_IQ3_S:
                    ಸ್ಥಿತಿ GGML_TYPE_IQ4_XS:
                    ಸ್ಥಿತಿ GGML_TYPE_IQ4_NL:
                    ಸ್ಥಿತಿ GGML_TYPE_MXFP4:
                    ಸ್ಥಿತಿ GGML_TYPE_NVFP4:
                    ಸ್ಥಿತಿ GGML_TYPE_TQ1_0:
                    ಸ್ಥಿತಿ GGML_TYPE_TQ2_0:
                    ಸ್ಥಿತಿ GGML_TYPE_I32:
                        ಹಿಂತಿರುಗಿ ನಿಜ;
                    ಪೂರ್ವನಿಯೋಜಿತ:
                        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
                }
            }
        ಸ್ಥಿತಿ GGML_OP_GET_ಅಡ್ಡಸಾಲುಗಳು_ಹಿಂಭಾಗ:
            ಹಿಂತಿರುಗಿ op->type == GGML_TYPE_F32 && op->src[0]->type == GGML_TYPE_F32;
        ಸ್ಥಿತಿ GGML_OP_ಗಣ_ಅಡ್ಡಸಾಲುಗಳು:
            {
                ಆದರೆ ((op->src[0]->type != GGML_TYPE_F32 && op->src[0]->type != GGML_TYPE_F16) ||
                    (op->src[1]->type != GGML_TYPE_I32 && op->src[1]->type != GGML_TYPE_I64)) {
                    ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
                }
                ಆಯ್ಕೆ (op->type) {
                    ಸ್ಥಿತಿ GGML_TYPE_F32:
                    ಸ್ಥಿತಿ GGML_TYPE_F16:
                    ಸ್ಥಿತಿ GGML_TYPE_BF16:
                    ಸ್ಥಿತಿ GGML_TYPE_Q1_0:
                    ಸ್ಥಿತಿ GGML_TYPE_Q2_0:
                    ಸ್ಥಿತಿ GGML_TYPE_Q4_0:
                    ಸ್ಥಿತಿ GGML_TYPE_Q4_1:
                    ಸ್ಥಿತಿ GGML_TYPE_Q5_0:
                    ಸ್ಥಿತಿ GGML_TYPE_Q5_1:
                    ಸ್ಥಿತಿ GGML_TYPE_Q8_0:
                    ಸ್ಥಿತಿ GGML_TYPE_IQ4_NL:
                        ಹಿಂತಿರುಗಿ ನಿಜ;
                    ಪೂರ್ವನಿಯೋಜಿತ:
                        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
                }
            }
        ಸ್ಥಿತಿ GGML_OP_CONT:
        ಸ್ಥಿತಿ GGML_OP_CPY:
        ಸ್ಥಿತಿ GGML_OP_DUP:
            {
                ggml_type src0_type = op->src[0]->type;
                ggml_type src1_type = op->src[1] != ಶೂನ್ಯಸೂಚಕ ? op->src[1]->type : src0_type;

                ಆದರೆ (src0_type == GGML_TYPE_F32) {
                    ಆಯ್ಕೆ (src1_type) {
                    ಸ್ಥಿತಿ GGML_TYPE_F32:
                    ಸ್ಥಿತಿ GGML_TYPE_F16:
                    ಸ್ಥಿತಿ GGML_TYPE_BF16:
                    ಸ್ಥಿತಿ GGML_TYPE_Q1_0:
                    ಸ್ಥಿತಿ GGML_TYPE_Q2_0:
                    ಸ್ಥಿತಿ GGML_TYPE_Q4_0:
                    ಸ್ಥಿತಿ GGML_TYPE_Q4_1:
                    ಸ್ಥಿತಿ GGML_TYPE_Q5_0:
                    ಸ್ಥಿತಿ GGML_TYPE_Q5_1:
                    ಸ್ಥಿತಿ GGML_TYPE_Q8_0:
                    ಸ್ಥಿತಿ GGML_TYPE_IQ4_NL:
                        ಹಿಂತಿರುಗಿ ನಿಜ;
                    ಪೂರ್ವನಿಯೋಜಿತ:
                        ನಿಲ್ಲಿಸು;
                    }
                }
                ಆದರೆ (src1_type == GGML_TYPE_F32) {
                    ಆಯ್ಕೆ (src0_type) {
                    ಸ್ಥಿತಿ GGML_TYPE_F16:
                    ಸ್ಥಿತಿ GGML_TYPE_BF16:
                    ಸ್ಥಿತಿ GGML_TYPE_Q1_0:
                    ಸ್ಥಿತಿ GGML_TYPE_Q2_0:
                    ಸ್ಥಿತಿ GGML_TYPE_Q4_0:
                    ಸ್ಥಿತಿ GGML_TYPE_Q4_1:
                    ಸ್ಥಿತಿ GGML_TYPE_Q5_0:
                    ಸ್ಥಿತಿ GGML_TYPE_Q5_1:
                    ಸ್ಥಿತಿ GGML_TYPE_Q8_0:
                    ಸ್ಥಿತಿ GGML_TYPE_IQ4_NL:
                        ಹಿಂತಿರುಗಿ ನಿಜ;
                    ಪೂರ್ವನಿಯೋಜಿತ:
                        ನಿಲ್ಲಿಸು;
                    }
                }

                ಆದರೆ (src0_type == GGML_TYPE_F16 && src1_type == GGML_TYPE_F16) {
                    ಹಿಂತಿರುಗಿ ನಿಜ;
                }

                ಆದರೆ (
                    (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_I32) ||
                    (src0_type == GGML_TYPE_I32 && src1_type == GGML_TYPE_F32)
                ) {
                    ಹಿಂತಿರುಗಿ ನಿಜ;
                }

                // We can handle copying from a type to the same type if it's
                // either not quantized or is quantized and contiguous.
                // We use f16 or f32 shaders to do the copy,
                // so the type/block size must be a multiple of 4.
                ಆದರೆ (src0_type == src1_type &&
                    (!ggml_is_quantized(src0_type) || (ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op))) &&
                    (ggml_type_ಗಾತ್ರ(src0_type) % 2) == 0) {
                    ಹಿಂತಿರುಗಿ ನಿಜ;
                }
                ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
            }
        ಸ್ಥಿತಿ GGML_OP_REPEAT:
            ಹಿಂತಿರುಗಿ ggml_type_ಗಾತ್ರ(op->type) == ggml_type_ಗಾತ್ರ(op->src[0]->type) &&
                  (ggml_type_ಗಾತ್ರ(op->type) == ಗಾತ್ರ(ದಶಮಾಂಶ) || ggml_type_ಗಾತ್ರ(op->type) == 2);
        ಸ್ಥಿತಿ GGML_OP_REPEAT_ಹಿಂಭಾಗ:
            ಹಿಂತಿರುಗಿ op->type == GGML_TYPE_F32 && op->src[0]->type == GGML_TYPE_F32;
        ಸ್ಥಿತಿ GGML_OP_ROPE:
            ಹಿಂತಿರುಗಿ ggml_is_contiguous_ಅಡ್ಡಸಾಲುಗಳು(op) && ggml_is_contiguous_ಅಡ್ಡಸಾಲುಗಳು(op->src[0]);
        ಸ್ಥಿತಿ GGML_OP_ROPE_ಹಿಂಭಾಗ:
        ಸ್ಥಿತಿ GGML_OP_NONE:
        ಸ್ಥಿತಿ GGML_OP_RESHAPE:
        ಸ್ಥಿತಿ GGML_OP_VIEW:
        ಸ್ಥಿತಿ GGML_OP_ಕ್ರಮಬದಲಾವಣೆ:
        ಸ್ಥಿತಿ GGML_OP_TRANSPOSE:
        ಸ್ಥಿತಿ GGML_OP_RMS_NORM:
            ಹಿಂತಿರುಗಿ ನಿಜ;
        ಸ್ಥಿತಿ GGML_OP_ಗುಂಪು_NORM:
            ಹಿಂತಿರುಗಿ ggml_is_contiguous(op->src[0]);
        ಸ್ಥಿತಿ GGML_OP_NORM:
        ಸ್ಥಿತಿ GGML_OP_L2_NORM:
            ಹಿಂತಿರುಗಿ op->src[0]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32;
        ಸ್ಥಿತಿ GGML_OP_ಕೂಡಿಸು:
        ಸ್ಥಿತಿ GGML_OP_SUB:
        ಸ್ಥಿತಿ GGML_OP_MUL:
        ಸ್ಥಿತಿ GGML_OP_DIV:
            ಹಿಂತಿರುಗಿ (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16) &&
                   (op->src[1]->type == GGML_TYPE_F32 || op->src[1]->type == GGML_TYPE_F16) &&
                   (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16);
        ಸ್ಥಿತಿ GGML_OP_ಕೂಡಿಸು_ID:
            ಹಿಂತಿರುಗಿ op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32 && op->src[2]->type == GGML_TYPE_I32 &&
                   op->type == GGML_TYPE_F32;
        ಸ್ಥಿತಿ GGML_OP_SILU_ಹಿಂಭಾಗ:
        ಸ್ಥಿತಿ GGML_OP_RMS_NORM_ಹಿಂಭಾಗ:
            ಹಿಂತಿರುಗಿ ggml_is_contiguous(op->src[0]) && op->src[0]->type == GGML_TYPE_F32;
        ಸ್ಥಿತಿ GGML_OP_SQR:
        ಸ್ಥಿತಿ GGML_OP_SQRT:
        ಸ್ಥಿತಿ GGML_OP_SIN:
        ಸ್ಥಿತಿ GGML_OP_COS:
        ಸ್ಥಿತಿ GGML_OP_CLAMP:
        ಸ್ಥಿತಿ GGML_OP_LEAKY_RELU:
            ಹಿಂತಿರುಗಿ (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16) &&
                   op->type == op->src[0]->type;
        ಸ್ಥಿತಿ GGML_OP_OPT_ಹೆಜ್ಜೆ_ADAMW:
        ಸ್ಥಿತಿ GGML_OP_OPT_ಹೆಜ್ಜೆ_SGD:
            ಹಿಂತಿರುಗಿ ggml_is_contiguous(op->src[0]) && op->src[0]->type == GGML_TYPE_F32;
        ಸ್ಥಿತಿ GGML_OP_ಹೊರಹೋಗು_PROD:
            ಹಿಂತಿರುಗಿ ggml_is_contiguous(op->src[0]) && op->src[0]->type == GGML_TYPE_F32
                && ggml_is_contiguous(op->src[1]) && op->src[1]->type == GGML_TYPE_F32
                && op->type == GGML_TYPE_F32;
        ಸ್ಥಿತಿ GGML_OP_LOG:
        ಸ್ಥಿತಿ GGML_OP_TRI:
        ಸ್ಥಿತಿ GGML_OP_DIAG:
            ಹಿಂತಿರುಗಿ (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16) &&
                   op->type == op->src[0]->type;
        ಸ್ಥಿತಿ GGML_OP_ARGSORT:
            {
                ಆದರೆ (!ggml_is_contiguous(op) || !ggml_is_contiguous(op->src[0])) {
                    ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
                }
                // pipeline_argsort_large_f32 requires vulkan memory model.
                ಆದರೆ (device->vulkan_ಸ್ಮರಣೆ_model) {
                    ಹಿಂತಿರುಗಿ ನಿಜ;
                } ಇಲ್ಲದಿದ್ದರೆ {
                    ಹಿಂತಿರುಗಿ op->ne[0] <= (1 << std::min(device->max_workgroup_ಗಾತ್ರ_log2, ಅಂಕೆ_argsort_pipelines - 1));
                }
            }
        ಸ್ಥಿತಿ GGML_OP_ತುದಿ_K:
            {
                ಆದರೆ (!ggml_is_contiguous(op) || !ggml_is_contiguous(op->src[0])) {
                    ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
                }
                // large k falls back to radix-select
                ಸ್ಥಿರಾಂಕ uint32_t min_pipeline =
                    std::max((uint32_t) log2f(ದಶಮಾಂಶ(op->ne[0])) + 1, device->subgroup_ಗಾತ್ರ_log2);
                ಆದರೆ (min_pipeline < ಅಂಕೆ_topk_pipelines && device->pipeline_topk_f32[min_pipeline]) {
                    ಹಿಂತಿರುಗಿ ನಿಜ;
                }
                ಹಿಂತಿರುಗಿ device->pipeline_topk_radix_f32 != ಶೂನ್ಯಸೂಚಕ;
            }
        ಸ್ಥಿತಿ GGML_OP_UPSCALE:
            ಆದರೆ (op->op_params[0] & GGML_SCALE_FLAG_ANTIALIAS) {
                ಆದರೆ ((op->op_params[0] & 0xFF) != GGML_SCALE_ಬಹುಲಕ_BILINEAR) {
                    ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
                }
            }
            ಹಿಂತಿರುಗಿ op->src[0]->type == GGML_TYPE_F32;
        ಸ್ಥಿತಿ GGML_OP_ACC:
            ಹಿಂತಿರುಗಿ op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32;
        ಸ್ಥಿತಿ GGML_OP_ಗಣ:
            ಹಿಂತಿರುಗಿ op->src[0]->type == op->src[1]->type && op->src[0]->type == op->type &&
                   (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_I32);
        ಸ್ಥಿತಿ GGML_OP_CONCAT: {
            ಹಿಂತಿರುಗಿ ggml_vk_concat_supported(op->src[0], op->src[1], op);
        }
        ಸ್ಥಿತಿ GGML_OP_ADD1:
            ಹಿಂತಿರುಗಿ (op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32)
                || (op->src[0]->type == GGML_TYPE_F16 && op->src[1]->type == GGML_TYPE_F32)
                || (op->src[0]->type == GGML_TYPE_F16 && op->src[1]->type == GGML_TYPE_F16);
        ಸ್ಥಿತಿ GGML_OP_ARANGE:
            ಹಿಂತಿರುಗಿ op->type == GGML_TYPE_F32;
        ಸ್ಥಿತಿ GGML_OP_FILL:
            ಹಿಂತಿರುಗಿ op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16;
        ಸ್ಥಿತಿ GGML_OP_SCALE:
            ಹಿಂತಿರುಗಿ ggml_is_contiguous(op->src[0]) && op->src[0]->type == GGML_TYPE_F32;
        ಸ್ಥಿತಿ GGML_OP_PAD:
        ಸ್ಥಿತಿ GGML_OP_PAD_REFLECT_1D:
        ಸ್ಥಿತಿ GGML_OP_ROLL:
            ಹಿಂತಿರುಗಿ op->src[0]->type == GGML_TYPE_F32;
        ಸ್ಥಿತಿ GGML_OP_DIAG_ಮುಖವಾಡ_INF:
            ಹಿಂತಿರುಗಿ ggml_is_contiguous(op->src[0]) && op->src[0]->type == GGML_TYPE_F32;
        ಸ್ಥಿತಿ GGML_OP_SOFT_MAX:
            ಹಿಂತಿರುಗಿ ggml_is_contiguous(op->src[0]) && op->src[0]->type == GGML_TYPE_F32
                && (!op->src[1] || (op->src[1]->type == GGML_TYPE_F32 || op->src[1]->type == GGML_TYPE_F16));
        ಸ್ಥಿತಿ GGML_OP_SOFT_MAX_ಹಿಂಭಾಗ:
            ಹಿಂತಿರುಗಿ ggml_is_contiguous(op->src[0]) && op->src[0]->type == GGML_TYPE_F32
                && ggml_is_contiguous(op->src[1]) && op->src[1]->type == GGML_TYPE_F32;
        ಸ್ಥಿತಿ GGML_OP_ಮೊತ್ತ:
            ಹಿಂತಿರುಗಿ op->src[0]->type == GGML_TYPE_F32 && ggml_is_contiguously_allocated(op->src[0]);
        ಸ್ಥಿತಿ GGML_OP_ಮೊತ್ತ_ಅಡ್ಡಸಾಲುಗಳು:
        ಸ್ಥಿತಿ GGML_OP_ಸರಾಸರಿ:
            ಹಿಂತಿರುಗಿ op->src[0]->type == GGML_TYPE_F32 && ggml_is_contiguous_ಅಡ್ಡಸಾಲುಗಳು(op->src[0]);
        ಸ್ಥಿತಿ GGML_OP_CUMSUM:
            {
                ಆದರೆ (device->subgroup_arithmetic && device->subgroup_require_ತುಂಬಿದ_support) {
                    ಹಿಂತಿರುಗಿ op->src[0]->type == GGML_TYPE_F32 && ggml_is_contiguous_ಅಡ್ಡಸಾಲುಗಳು(op->src[0]);
                }
                ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
            }
        ಸ್ಥಿತಿ GGML_OP_DSV4_HC_COMB:
        ಸ್ಥಿತಿ GGML_OP_DSV4_HC_PRE:
        ಸ್ಥಿತಿ GGML_OP_DSV4_HC_POST:
            {
                ಆದರೆ (op->type != GGML_TYPE_F32) {
                    ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
                }
                ಗಾಗಿ (uint32_t i = 0; i < GGML_MAX_SRC; ++i) {
                    ಆದರೆ (op->src[i] && op->src[i]->type != GGML_TYPE_F32) {
                        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
                    }
                }
                // hc is hardcoded to 4 in the shaders. ggml only constrains it
                // to 4 for COMB, so PRE/POST have to be checked here.
                ಆದರೆ (op->op == GGML_OP_DSV4_HC_PRE && op->src[0]->ne[1] != 4) {
                    ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
                }
                ಆದರೆ (op->op == GGML_OP_DSV4_HC_POST && op->src[1]->ne[1] != 4) {
                    ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
                }
                ಆದರೆ (op->op == GGML_OP_DSV4_HC_COMB) {
                    ಹಿಂತಿರುಗಿ device->pipeline_dsv4_hc_comb_f32 != ಶೂನ್ಯಸೂಚಕ;
                }
                ಹಿಂತಿರುಗಿ ನಿಜ;
            }
        ಸ್ಥಿತಿ GGML_OP_SOLVE_TRI:
            {
                ಆದರೆ (op->type != GGML_TYPE_F32 || op->src[0]->type != GGML_TYPE_F32) {
                    ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
                }
                ಸ್ಥಿರಾಂಕ uint32_t N = op->src[0]->ne[0];
                ಸ್ಥಿರಾಂಕ uint32_t K = op->src[1]->ne[0];
                // K dimension limited to workgroup size
                ಆದರೆ (K > 1u << device->max_workgroup_ಗಾತ್ರ_log2) {
                    ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
                }
                ಸ್ಥಿರಾಂಕ uint32_t batch_N = device->properties.limits.max_Compute_Shared_ಸ್ಮರಣೆ_ಗಾತ್ರ / ((N + K) * ಗಾತ್ರ(ದಶಮಾಂಶ));

                ಆದರೆ (batch_N == 0) {
                    ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
                }
                ಹಿಂತಿರುಗಿ ನಿಜ;
            }
        ಸ್ಥಿತಿ GGML_OP_ARGMAX:
            ಹಿಂತಿರುಗಿ ggml_is_contiguous(op->src[0]) && op->src[0]->type == GGML_TYPE_F32;
        ಸ್ಥಿತಿ GGML_OP_CROSS_ENTROPY_LOSS:
            ಹಿಂತಿರುಗಿ ggml_is_contiguous(op->src[0]) && op->src[0]->type == GGML_TYPE_F32
                && ggml_is_contiguous(op->src[1]) && op->src[1]->type == GGML_TYPE_F32
                && ggml_are_same_shape(op->src[0], op->src[1])
                && ggml_is_contiguous(op) && ggml_is_scalar(op) && op->type == GGML_TYPE_F32;
        ಸ್ಥಿತಿ GGML_OP_CROSS_ENTROPY_LOSS_ಹಿಂಭಾಗ:
            ಹಿಂತಿರುಗಿ ggml_is_contiguous(op->src[0]) && op->src[0]->type == GGML_TYPE_F32 && ggml_is_scalar(op->src[0])
                && ggml_is_contiguous(op->src[1]) && op->src[1]->type == GGML_TYPE_F32
                && ggml_is_contiguous(op->src[2]) && op->src[2]->type == GGML_TYPE_F32
                && ggml_are_same_shape(op->src[1], op->src[2])
                && ggml_are_same_shape(op->src[1], op)
                && ggml_is_contiguous(op) && op->type == GGML_TYPE_F32;
        ಸ್ಥಿತಿ GGML_OP_ಎಣಿಕೆ_EQUAL:
            ಹಿಂತಿರುಗಿ ggml_is_contiguous(op->src[0]) && op->src[0]->type == GGML_TYPE_I32
                && ggml_is_contiguous(op->src[1]) && op->src[1]->type == GGML_TYPE_I32;
        ಸ್ಥಿತಿ GGML_OP_IM2_ಲಂಬಸಾಲು:
            ಹಿಂತಿರುಗಿ ggml_is_contiguous(op->src[1])
                && op->src[1]->type == GGML_TYPE_F32
                && (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16);
        ಸ್ಥಿತಿ GGML_OP_IM2_ಲಂಬಸಾಲು_3_D:
            ಹಿಂತಿರುಗಿ op->src[1]->type == GGML_TYPE_F32
                && (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16);
        ಸ್ಥಿತಿ GGML_OP_TIMESTEP_EMBEDDING:
            ಹಿಂತಿರುಗಿ op->src[0]->type == GGML_TYPE_F32;
        ಸ್ಥಿತಿ GGML_OP_CONV_2D_DW:
            ಹಿಂತಿರುಗಿ (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16)
                && op->src[1]->type == GGML_TYPE_F32;
        ಸ್ಥಿತಿ GGML_OP_POOL_1D:
            ಹಿಂತಿರುಗಿ ggml_is_contiguous(op->src[0]) && op->src[0]->type == GGML_TYPE_F32;
        ಸ್ಥಿತಿ GGML_OP_POOL_2D:
            ಹಿಂತಿರುಗಿ ggml_is_contiguous(op->src[0]) && op->src[0]->type == GGML_TYPE_F32;
        ಸ್ಥಿತಿ GGML_OP_RWKV_WKV6:
        ಸ್ಥಿತಿ GGML_OP_RWKV_WKV7:
            ಹಿಂತಿರುಗಿ ನಿಜ; // all inputs are contiguous, see ggml.c
        ಸ್ಥಿತಿ GGML_OP_GATED_ರೇಖೀಯ_ATTN:
            // the shader block size is hardcoded to head_size 64
            ಹಿಂತಿರುಗಿ op->src[0]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32 && op->src[0]->ne[0] == 64;
        ಸ್ಥಿತಿ GGML_OP_LIGHTNING_INDEXER:
            {
                ಸ್ಥಿರಾಂಕ ggml_tensor * q = op->src[0];
                ಸ್ಥಿರಾಂಕ ggml_tensor * k = op->src[1];
                ಸ್ಥಿರಾಂಕ ggml_tensor * w = op->src[2];
                ಸ್ಥಿರಾಂಕ ggml_tensor * m = op->src[3];

                // the q/w/m types and the shape relationships between q, k, w, m and dst
                // are already asserted in ggml_lightning_indexer()
                ಆದರೆ (!ggml_vk_lightning_indexer_k_type_supported(k->type) || !device->fp16) {
                    ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
                }

                // the shader block size is hardcoded to head size 128
                ಆದರೆ (q->ne[0] != 128) {
                    ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
                }

                // the shader indexes the buffers by element stride, and is dispatched
                // without allow_misalign
                ಗಾಗಿ (ಸ್ಥಿರಾಂಕ ggml_tensor * t : {q, k, w, m, op}) {
                    ಆದರೆ (t->nb[0] != ggml_type_ಗಾತ್ರ(t->type) ||
                        (vk_tensor_ಸ್ಥಾನಾಂತರ(t) + t->view_offs) % device->properties.limits.min_Storage_ಬಫರ್_ಸ್ಥಾನಾಂತರ_Alignment != 0) {
                        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
                    }
                    // the strides get scaled down from bytes, so the division must be exact
                    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 1; i < GGML_MAX_DIMS; ++i) {
                        ಆದರೆ (t->nb[i] % ggml_type_ಗಾತ್ರ(t->type) != 0) {
                            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
                        }
                    }
                }
                ಹಿಂತಿರುಗಿ ನಿಜ;
            }
        ಸ್ಥಿತಿ GGML_OP_GATED_DELTA_NET:
            {
                ಸ್ಥಿರಾಂಕ uint32_t S_v = op->src[2]->ne[0];
                ಆದರೆ (S_v != 16 && S_v != 32 && S_v != 64 && S_v != 128) {
                    ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
                }
                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < 6; i++) {
                    ಆದರೆ (op->src[i] == ಶೂನ್ಯಸೂಚಕ || op->src[i]->type != GGML_TYPE_F32) {
                        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
                    }
                }
                ಹಿಂತಿರುಗಿ op->type == GGML_TYPE_F32;
            }
        ಸ್ಥಿತಿ GGML_OP_SSM_ಪರಿಶೀಲನೆ:
            {
                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < 6; i++) {
                    ಆದರೆ (op->src[i] && ggml_is_quantized(op->src[i]->type)) {
                        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
                    }
                }
                ಆದರೆ (op->src[6] && op->src[6]->type != GGML_TYPE_I32) {
                    ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
                }
                ಆದರೆ (op->src[0]->type != GGML_TYPE_F32 || op->type != GGML_TYPE_F32) {
                    ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
                }

                ಸ್ಥಿರಾಂಕ uint32_t d_state = op->src[0]->ne[0];
                ಸ್ಥಿರಾಂಕ uint32_t ತಲೆ_dim = op->src[0]->ne[1];

                ತಾರ್ಕಿಕ is_mamba2 = (op->src[3] && op->src[3]->nb[1] == ಗಾತ್ರ(ದಶಮಾಂಶ));
                ಆದರೆ (!is_mamba2) {
                    ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
                }

                ಆದರೆ ((d_state != 128 && d_state != 256) || ತಲೆ_dim % 16 != 0) {
                    ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
                }

                size_t shmem_ಗಾತ್ರ = d_state * ಗಾತ್ರ(ದಶಮಾಂಶ);

                ಆದರೆ (shmem_ಗಾತ್ರ > device->properties.limits.max_Compute_Shared_ಸ್ಮರಣೆ_ಗಾತ್ರ) {
                    ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
                }

                ಆದರೆ (!device->subgroup_basic) {
                    ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
                }

                ಹಿಂತಿರುಗಿ ನಿಜ;
            }
        ಸ್ಥಿತಿ GGML_OP_SSM_CONV:
            ಹಿಂತಿರುಗಿ op->src[0]->type == GGML_TYPE_F32;
        ಸ್ಥಿತಿ GGML_OP_CONV_TRANSPOSE_1D:
            ಹಿಂತಿರುಗಿ op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32;
        ಸ್ಥಿತಿ GGML_OP_COL2IM_1D:
            ಹಿಂತಿರುಗಿ (op->src[0]->type == GGML_TYPE_F32 ||
                    op->src[0]->type == GGML_TYPE_F16 ||
                    op->src[0]->type == GGML_TYPE_BF16) &&
                   op->type == op->src[0]->type &&
                   ggml_is_contiguous(op->src[0]) &&
                   ggml_is_contiguous(op);
        ಸ್ಥಿತಿ GGML_OP_CONV_2D:
        ಸ್ಥಿತಿ GGML_OP_CONV_TRANSPOSE_2D:
            {
                ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ transpose = op->op == GGML_OP_CONV_TRANSPOSE_2D;
                ಸ್ಥಿರಾಂಕ int64_t ಹೊರಗೆ = !transpose ? op->src[0]->ne[3] : op->src[0]->ne[2];
                ಸ್ಥಿರಾಂಕ int64_t ಒಳಗೆ  = !transpose ? op->src[0]->ne[2] : op->src[0]->ne[3];

                // Channel-contiguous format is not supported yet.
                ಹಿಂತಿರುಗಿ ((op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16) &&
                    (op->src[0]->nb[0] == ಗಾತ್ರ(ದಶಮಾಂಶ) || op->src[0]->nb[0] == ಗಾತ್ರ(ggml_fp16_t) ) &&
                    op->src[1]->type == GGML_TYPE_F32 &&
                    op->type == GGML_TYPE_F32 &&
                    ಹೊರಗೆ == op->ne[2] &&
                    ಒಳಗೆ == op->src[1]->ne[2] &&
                    ggml_is_contiguous(op->src[0]) &&
                    ggml_is_contiguous(op->src[1]) &&
                    ggml_is_contiguous(op));
            }
        ಸ್ಥಿತಿ GGML_OP_CONV_3D:
            ಹಿಂತಿರುಗಿ (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16) &&
                op->src[1]->type == GGML_TYPE_F32 &&
                op->type == GGML_TYPE_F32 &&
                ggml_is_contiguous(op->src[0]) &&
                ggml_is_contiguous(op->src[1]) &&
                ggml_is_contiguous(op);
        ಪೂರ್ವನಿಯೋಜಿತ:
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಬಳಸದ(dev);
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_backend_vk_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_ಬಫರ್_type_t buft) {
    ಆದರೆ (buft->iface.get_name != ggml_backend_vk_ಬಫರ್_type_name) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ggml_backend_vk_device_context * ctx = (ggml_backend_vk_device_context *)dev->context;
    ggml_backend_vk_ಬಫರ್_type_context * buft_ctx = (ggml_backend_vk_ಬಫರ್_type_context *)buft->context;

    ಹಿಂತಿರುಗಿ buft_ctx->device->idx == ctx->device;
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_backend_vk_device_offload_op(ggml_backend_dev_t dev, ಸ್ಥಿರಾಂಕ ggml_tensor * op) {
    ggml_backend_vk_device_context * dev_ctx = (ggml_backend_vk_device_context *)dev->context;

    ಹಿಂತಿರುಗಿ ggml_vk_get_op_batch_ಗಾತ್ರ(op) >= dev_ctx->op_offload_min_batch_ಗಾತ್ರ;
}

ಸ್ಥಿರ ggml_backend_event_t ggml_backend_vk_device_event_new(ggml_backend_dev_t dev) {
    ggml_backend_vk_device_context * ctx = (ggml_backend_vk_device_context *)dev->context;
    ಸ್ವಯಂಚಾಲಿತ device = ggml_vk_get_device(ctx->device);

    vk_event *vkev = ಹೊಸ vk_event;
    ಆದರೆ (!vkev) {
        ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
    }

    // No events initially, they get created on demand
    vkev->has_event = ಸುಳ್ಳು;

    vk::SemaphoreTypeCreateInfo tci{ vk::SemaphoreType::eTimeline, 0 };
    vk::SemaphoreCreateInfo ci{};
    ci.ಗಣ_PNext(&tci);
    vkev->tl_semaphore = { device->device.createSemaphore(ci), 0 };

    ಹಿಂತಿರುಗಿ ಹೊಸ ggml_backend_event {
        /* .device  = */ dev,
        /* .context = */ vkev,
    };
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_vk_device_event_free(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    ggml_backend_vk_device_context * ctx = (ggml_backend_vk_device_context *)dev->context;
    ಸ್ವಯಂಚಾಲಿತ device = ggml_vk_get_device(ctx->device);

    vk_event *vkev = (vk_event *)event->context;

    device->device.destroySemaphore(vkev->tl_semaphore.s);
    ಗಾಗಿ (ಸ್ವಯಂಚಾಲಿತ& event : vkev->events_free) {
        device->device.destroyEvent(event);
    }
    ಗಾಗಿ (ಸ್ವಯಂಚಾಲಿತ& event : vkev->events_submitted) {
        device->device.destroyEvent(event);
    }
    ಆದರೆ (vkev->has_event) {
        device->device.destroyEvent(vkev->event);
    }
    ಅಳಿಸು vkev;
    ಅಳಿಸು event;
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_vk_device_event_synchronize(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    VK_LOG_DEBUG("ggml_backend_vk_device_event_synchronize(backend=" << dev << ", event=" << event << ")");
    ggml_backend_vk_device_context * ctx = (ggml_backend_vk_device_context *)dev->context;
    ಸ್ವಯಂಚಾಲಿತ device = ggml_vk_get_device(ctx->device);
    vk_event *vkev = (vk_event *)event->context;

    // Only do something if the event has actually been used
    ಆದರೆ (vkev->has_event) {
        vk::Semaphore sem = vkev->tl_semaphore.s;
        uint64_t val = vkev->tl_semaphore.value;
        vk::SemaphoreWaitInfo swi{vk::SemaphoreWaitFlags{}, sem, val};
        VK_CHECK(device->device.waitSemaphores(swi, UINT64_MAX), "event_synchronize", device);

        // Reset and move submitted events
        ಗಾಗಿ (ಸ್ವಯಂಚಾಲಿತ& event : vkev->events_submitted) {
            device->device.resetEvent(event);
        }
        vkev->events_free.insert(vkev->events_free.end(), vkev->events_submitted.begin(), vkev->events_submitted.end());
        vkev->events_submitted.clear();

        // Finished using current command buffer so we flag for reuse
        ಆದರೆ (vkev->cmd_ಬಫರ್) {
            // Only flag for reuse if it hasn't been reused already
            ಆದರೆ (vkev->cmd_ಬಫರ್_use_counter == vkev->cmd_ಬಫರ್->use_counter) {
                vkev->cmd_ಬಫರ್->ಒಳ_use = ಸುಳ್ಳು;
                vkev->cmd_ಬಫರ್->buf.reset();
            }
            vkev->cmd_ಬಫರ್ = ಶೂನ್ಯಸೂಚಕ;
        }
    }
}

ಸ್ಥಿರ ggml_backend_ಬಫರ್_t ggml_backend_vk_device_ಬಫರ್_from_host_ptr(ggml_backend_dev_t dev, ಶೂನ್ಯ * ptr, size_t size, size_t max_tensor_ಗಾತ್ರ) {
    VK_LOG_DEBUG("ggml_backend_vk_device_buffer_from_host_ptr(backend=" << dev << ", ptr=" << ptr << ", size=" << size << ")");
    GGML_ಬಳಸದ(max_tensor_ಗಾತ್ರ);

    ggml_backend_vk_device_context * ctx = (ggml_backend_vk_device_context *)dev->context;
    ಸ್ವಯಂಚಾಲಿತ device = ggml_vk_get_device(ctx->device);

    vk_ಬಫರ್ buf = ggml_vk_ಬಫರ್_from_host_ptr(device, ptr, size);

    ಆದರೆ (!buf) {
        ಹಿಂತಿರುಗಿ {};
    }

    ggml_backend_vk_ಬಫರ್_context * bufctx = ಹೊಸ ggml_backend_vk_ಬಫರ್_context(device, std::move(buf), device->name);

    ggml_backend_ಬಫರ್_t ret = ggml_backend_ಬಫರ್_init(ggml_backend_vk_device_get_ಬಫರ್_type(dev), ggml_backend_vk_ಬಫರ್_interface, bufctx, size);

    ಹಿಂತಿರುಗಿ ret;
}

ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_backend_device_i ggml_backend_vk_device_i = {
    /* .get_name             = */ ggml_backend_vk_device_get_name,
    /* .get_description      = */ ggml_backend_vk_device_get_description,
    /* .get_memory           = */ ggml_backend_vk_device_get_ಸ್ಮರಣೆ,
    /* .get_type             = */ ggml_backend_vk_device_get_type,
    /* .get_props            = */ ggml_backend_vk_device_get_props,
    /* .init_backend         = */ ggml_backend_vk_device_init,
    /* .get_buffer_type      = */ ggml_backend_vk_device_get_ಬಫರ್_type,
    /* .get_host_buffer_type = */ ggml_backend_vk_device_get_host_ಬಫರ್_type,
    /* .buffer_from_host_ptr = */ ggml_backend_vk_device_ಬಫರ್_from_host_ptr,
    /* .supports_op          = */ ggml_backend_vk_device_supports_op,
    /* .supports_buft        = */ ggml_backend_vk_device_supports_buft,
    /* .offload_op           = */ ggml_backend_vk_device_offload_op,
    /* .event_new            = */ ggml_backend_vk_device_event_new,
    /* .event_free           = */ ggml_backend_vk_device_event_free,
    /* .event_synchronize    = */ ggml_backend_vk_device_event_synchronize,
};

ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * ggml_backend_vk_reg_get_name(ggml_backend_reg_t reg) {
    ಬಳಸದ(reg);
    ಹಿಂತಿರುಗಿ GGML_VK_NAME;
}

ಸ್ಥಿರ size_t ggml_backend_vk_reg_get_device_ಎಣಿಕೆ(ggml_backend_reg_t reg) {
    ಬಳಸದ(reg);
    ಹಿಂತಿರುಗಿ ggml_backend_vk_get_device_ಎಣಿಕೆ();
}

ಸ್ಥಿರ ggml_backend_dev_t ggml_backend_vk_reg_get_device(ggml_backend_reg_t reg, size_t device) {
    ಸ್ಥಿರ std::vector<ggml_backend_dev_t> devices;

    ಸ್ಥಿರ ತಾರ್ಕಿಕ initialized = ಸುಳ್ಳು;

    {
        ಸ್ಥಿರ std::mutex mutex;
        std::lock_guard<std::mutex> ಬೀಗ(mutex);
        ಆದರೆ (!initialized) {
            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ min_batch_ಗಾತ್ರ = getenv("GGML_OP_OFFLOAD_MIN_BATCH") ? atoi(getenv("GGML_OP_OFFLOAD_MIN_BATCH")) : 32;
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < ggml_backend_vk_get_device_ಎಣಿಕೆ(); i++) {
                ggml_backend_vk_device_context * ctx = ಹೊಸ ggml_backend_vk_device_context;
                ಅಕ್ಷರ desc[256];
                ggml_backend_vk_get_device_description(i, desc, ಗಾತ್ರ(desc));
                ctx->device = i;
                ctx->name = GGML_VK_NAME + std::to_string(i);
                ctx->description = desc;
                ctx->is_integrated_gpu = ggml_backend_vk_get_device_type(i) == vk::PhysicalDeviceType::eIntegratedGpu;
                ctx->pci_bus_id = ggml_backend_vk_get_device_pci_id(i);
                ctx->op_offload_min_batch_ಗಾತ್ರ = min_batch_ಗಾತ್ರ;
                devices.push_back(ಹೊಸ ggml_backend_device {
                    /* .iface   = */ ggml_backend_vk_device_i,
                    /* .reg     = */ reg,
                    /* .context = */ ctx,
                });
            }
            initialized = ನಿಜ;
        }
    }

    GGML_ASSERT(device < devices.size());
    ಹಿಂತಿರುಗಿ devices[device];
}

ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_backend_reg_i ggml_backend_vk_reg_i = {
    /* .get_name         = */ ggml_backend_vk_reg_get_name,
    /* .get_device_count = */ ggml_backend_vk_reg_get_device_ಎಣಿಕೆ,
    /* .get_device       = */ ggml_backend_vk_reg_get_device,
    /* .get_proc_address = */ ಖಾಲಿ,
};

ggml_backend_reg_t ggml_backend_vk_reg() {
    ಸ್ಥಿರ ggml_backend_reg reg = {
        /* .api_version = */ GGML_BACKEND_API_ಆವೃತ್ತಿ,
        /* .iface       = */ ggml_backend_vk_reg_i,
        /* .context     = */ ಶೂನ್ಯಸೂಚಕ,
    };
    ಪ್ರಯತ್ನಿಸು {
        ggml_vk_instance_init();
        ಹಿಂತಿರುಗಿ &reg;
    } ಹಿಡಿ (ಸ್ಥಿರಾಂಕ vk::SystemError& e) {
        VK_LOG_DEBUG("ggml_backend_vk_reg() -> Error: System error: " << e.what());
        ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
    } ಹಿಡಿ (ಸ್ಥಿರಾಂಕ std::exception &e) {
        VK_LOG_DEBUG("ggml_backend_vk_reg() -> Error: " << e.what());
        ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
    } ಹಿಡಿ (...) {
        VK_LOG_DEBUG("ggml_backend_vk_reg() -> Error: unknown exception during Vulkan init");
        ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
    }
}

ತಾರ್ಕಿಕ ggml_vk_instance_layer_settings_ಲಭ್ಯ() {
#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_VULKAN_VALIDATE
    // Check if validation layer provides the extension
    ಸ್ಥಿರಾಂಕ std::string layer_name = "VK_LAYER_KHRONOS_validation";
    ಗಾಗಿ (ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ& layer : vk::enumerateInstanceLayerProperties()) {
        ಆದರೆ (layer_name == layer.layerName.ದತ್ತಾಂಶ()) {
            ಗಾಗಿ (ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ& ext : vk::enumerateInstanceExtensionProperties(layer_name)) {
                ಆದರೆ (strcmp("VK_EXT_layer_settings", ext.extensionName.ದತ್ತಾಂಶ()) == 0) {
                    ಹಿಂತಿರುಗಿ ನಿಜ;
                }
            }
        }
    }

    std::cerr << "ggml_vulkan: WARNING: Validation layer or layer extension VK_EXT_layer_settings not found." << std::ಸಾಲು_ಅಂತ್ಯ;
#ಆದರೆ_ಅಂತ್ಯ
    ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
}

ತಾರ್ಕಿಕ ggml_vk_instance_portability_enumeration_ext_ಲಭ್ಯ(ಸ್ಥಿರಾಂಕ std::vector<vk::ExtensionProperties>& instance_extensions) {
#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ __APPLE__
    // Check for portability enumeration extension for MoltenVK support
    ಗಾಗಿ (ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ& properties : instance_extensions) {
        ಆದರೆ (strcmp("VK_KHR_portability_enumeration", properties.extensionName) == 0) {
            ಹಿಂತಿರುಗಿ ನಿಜ;
        }
    }
    std::cerr << "ggml_vulkan: WARNING: Instance extension VK_KHR_portability_enumeration not found." << std::ಸಾಲು_ಅಂತ್ಯ;
#ಆದರೆ_ಅಂತ್ಯ
    ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;

    ಬಳಸದ(instance_extensions);
}

ತಾರ್ಕಿಕ ggml_vk_instance_debug_utils_ext_ಲಭ್ಯ(
    ಸ್ಥಿರಾಂಕ std::vector<vk::ExtensionProperties> & instance_extensions) {
    // Check for portability enumeration extension for MoltenVK support
    ಗಾಗಿ (ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ & properties : instance_extensions) {
        ಆದರೆ (strcmp("VK_EXT_debug_utils", properties.extensionName) == 0) {
            ಹಿಂತಿರುಗಿ ನಿಜ;
        }
    }

    std::cerr << "ggml_vulkan: WARNING: Instance extension VK_EXT_debug_utils not found." << std::ಸಾಲು_ಅಂತ್ಯ;
    ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;

    ಬಳಸದ(instance_extensions);
}

ತಾರ್ಕಿಕ ggml_vk_device_is_supported(ಸ್ಥಿರಾಂಕ vk::PhysicalDevice & vkdev) {
    VkPhysicalDeviceFeatures2 device_features2;
    device_features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;

    VkPhysicalDeviceVulkan11Features vk11_features;
    vk11_features.p_ಮುಂದಿನ = ಶೂನ್ಯಸೂಚಕ;
    vk11_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
    device_features2.p_ಮುಂದಿನ = &vk11_features;

    vkGetPhysicalDeviceFeatures2(vkdev, &device_features2);

    ಹಿಂತಿರುಗಿ vk11_features.storage_Buffer16_ಬಿಟ್_Access;
}

ತಾರ್ಕಿಕ ggml_vk_khr_cooperative_ಮಾತೃಕೆ_support(ಸ್ಥಿರಾಂಕ vk::PhysicalDeviceProperties& props, ಸ್ಥಿರಾಂಕ vk::PhysicalDeviceDriverProperties& driver_props, vk_device_architecture arch) {
    ಆಯ್ಕೆ (props.vendorID) {
    ಸ್ಥಿತಿ VK_VENDOR_ID_INTEL:
        // Only allowing Xe2/Xe3 GPU and integrated Xe GPUs at the moment since older hardware (ex. Arc A770) has performance regressions.
        ಹಿಂತಿರುಗಿ (arch == vk_device_architecture::INTEL_XE2) ||
            (arch == vk_device_architecture::INTEL_XE1 && props.deviceType == vk::PhysicalDeviceType::eIntegratedGpu && driver_props.driverID == vk::DriverId::eIntelProprietaryWindows);
    ಸ್ಥಿತಿ VK_VENDOR_ID_AMD:
        ಆದರೆ (driver_props.driverID == vk::DriverId::eAmdProprietary || driver_props.driverID == vk::DriverId::e_Amd_Open_ಮೂಲ) {
            // Workaround for AMD proprietary driver reporting support on all GPUs
            ಹಿಂತಿರುಗಿ arch == vk_device_architecture::AMD_RDNA3 || arch == vk_device_architecture::AMD_RDNA4;
        }
        ಹಿಂತಿರುಗಿ ನಿಜ;
    ಸ್ಥಿತಿ VK_VENDOR_ID_QUALCOMM:
        // Only allow Adreno GPUs with hardware matrix cores (Gen 6+).
        ಹಿಂತಿರುಗಿ arch == vk_device_architecture::QUALCOMM_ADRENO;
    ಪೂರ್ವನಿಯೋಜಿತ:
        ಹಿಂತಿರುಗಿ ನಿಜ;
    }
}

uint32_t ggml_vk_intel_shader_core_ಎಣಿಕೆ(ಸ್ಥಿರಾಂಕ vk::PhysicalDevice& vkdev) {
    VkPhysicalDeviceProperties2 props = vkdev.getProperties2();

    ಆದರೆ (props.properties.vendorID != VK_VENDOR_ID_INTEL) {
        ಹಿಂತಿರುಗಿ 0;
    }

    ಸ್ಥಿರಾಂಕ uint32_t device_id = props.properties.deviceID;

    ಆಯ್ಕೆ (device_id) {
    ಸ್ಥಿತಿ 0x56A6:  // A310
        ಹಿಂತಿರುಗಿ 6;
    ಸ್ಥಿತಿ 0x5693:  // A370M
    ಸ್ಥಿತಿ 0x56A5:  // A380
    ಸ್ಥಿತಿ 0x56B1:  // Pro A40/A50
        ಹಿಂತಿರುಗಿ 8;
    ಸ್ಥಿತಿ 0x5697:  // A530M
        ಹಿಂತಿರುಗಿ 12;
    ಸ್ಥಿತಿ 0x5692:  // A550M
    ಸ್ಥಿತಿ 0x56B3:  // Pro A60
        ಹಿಂತಿರುಗಿ 16;
    ಸ್ಥಿತಿ 0x56A2:  // A580
        ಹಿಂತಿರುಗಿ 24;
    ಸ್ಥಿತಿ 0x5691:  // A730M
    ಸ್ಥಿತಿ 0x56A1:  // A750
        ಹಿಂತಿರುಗಿ 28;
    ಸ್ಥಿತಿ 0x56A0:  // A770
    ಸ್ಥಿತಿ 0x5690:  // A770M
        ಹಿಂತಿರುಗಿ 32;
    ಸ್ಥಿತಿ 0xE212:  // Pro B50
        ಹಿಂತಿರುಗಿ 16;
    ಸ್ಥಿತಿ 0xE20C:  // B570
        ಹಿಂತಿರುಗಿ 18;
    ಸ್ಥಿತಿ 0xE20B:  // B580
    ಸ್ಥಿತಿ 0xE211:  // Pro B60
    ಸ್ಥಿತಿ 0xE222:  // Pro B65
        ಹಿಂತಿರುಗಿ 20;
    ಸ್ಥಿತಿ 0xE223:  // Pro B70
        ಹಿಂತಿರುಗಿ 32;
    ಸ್ಥಿತಿ 0xB080:  // PTL Xe3 LPG 2x6 (12 subslices)
        ಹಿಂತಿರುಗಿ 12;
    ಪೂರ್ವನಿಯೋಜಿತ:
        ಹಿಂತಿರುಗಿ 0;
    }
}

ತಾರ್ಕಿಕ ggml_vk_intel_windows_driver_ಒಳ_range(uint32_t driver_ಆವೃತ್ತಿ, uint32_t ಅಧೋ_major, uint32_t ಅಧೋ_minor, uint32_t ಊರ್ಧ್ವ_major, uint32_t ಊರ್ಧ್ವ_minor) {
#ಆದರೆ defined(_WIN32)
    // Intel Windows encodes xxx.yyyy as [31:14].[13:0].
    ಸ್ಥಿರಾಂಕ uint32_t major = driver_ಆವೃತ್ತಿ >> 14;
    ಸ್ಥಿರಾಂಕ uint32_t minor = driver_ಆವೃತ್ತಿ & 0x3fff;

    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ ge_ಅಧೋ = major > ಅಧೋ_major || (major == ಅಧೋ_major && minor >= ಅಧೋ_minor);
    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ lt_ಊರ್ಧ್ವ = major < ಊರ್ಧ್ವ_major || (major == ಊರ್ಧ್ವ_major && minor < ಊರ್ಧ್ವ_minor);

    ಹಿಂತಿರುಗಿ ge_ಅಧೋ && lt_ಊರ್ಧ್ವ;
#ಇಲ್ಲದಿದ್ದರೆ
    GGML_ಬಳಸದ(driver_ಆವೃತ್ತಿ);
    GGML_ಬಳಸದ(ಅಧೋ_major);
    GGML_ಬಳಸದ(ಅಧೋ_minor);
    GGML_ಬಳಸದ(ಊರ್ಧ್ವ_major);
    GGML_ಬಳಸದ(ಊರ್ಧ್ವ_minor);
    ಹಿಂತಿರುಗಿ ನಿಜ;
#ಆದರೆ_ಅಂತ್ಯ
}

GGML_BACKEND_DL_IMPL(ggml_backend_vk_reg)


// out-of-lined header method definitions

ಶೂನ್ಯ vk_ಸರತಿ_handle_synchronized::submit(vk::ArrayProxy<ಸ್ಥಿರಾಂಕ vk::SubmitInfo> submits, vk::Fence fence) {
    // Workaround for NVIDIA driver bug
    std::unique_lock<std::mutex> device_guard;
    ಆದರೆ (device_submit_mutex) {
        device_guard = std::unique_lock<std::mutex>(*device_submit_mutex);
    }
    std::lock_guard<std::mutex> guard(mutex);
    ಪ್ರಯತ್ನಿಸು {
        queue.submit(submits, fence);
    } ಹಿಡಿ (vk::Device_ಕಳೆದುಹೋದ_Error &) {
        ಆದರೆ (ಸ್ವಯಂಚಾಲಿತ dev = device.ಬೀಗ()) {
            ggml_vk_print_device_ಕಳೆದುಹೋದ_info(dev);
        }
        ಎಸೆ;
    }
}

ಶೂನ್ಯ vk_ಸರತಿ_handle_unsynchronized::submit(vk::ArrayProxy<ಸ್ಥಿರಾಂಕ vk::SubmitInfo> submits, vk::Fence fence) {
    // Workaround for NVIDIA driver bug
    std::unique_lock<std::mutex> device_guard;
    ಆದರೆ (device_submit_mutex) {
        device_guard = std::unique_lock<std::mutex>(*device_submit_mutex);
    }
    ಪ್ರಯತ್ನಿಸು {
        queue.submit(submits, fence);
    } ಹಿಡಿ (vk::Device_ಕಳೆದುಹೋದ_Error &) {
        ಆದರೆ (ಸ್ವಯಂಚಾಲಿತ dev = device.ಬೀಗ()) {
            ggml_vk_print_device_ಕಳೆದುಹೋದ_info(dev);
        }
        ಎಸೆ;
    }
}

vk_device_struct::~vk_device_struct() {
    // the build failed before the device was created, nothing to destroy
    ಆದರೆ (device == VK_NULL_HANDLE) {
        ಹಿಂತಿರುಗಿ;
    }

    VK_LOG_DEBUG("destroy device " << name);

    device.destroyFence(fence);

    ggml_vk_destroy_ಬಫರ್(sync_staging);

    ಆದರೆ (compute_ಸರತಿ) compute_ಸರತಿ->cmd_pool.destroy(device);
    ಆದರೆ (transfer_ಸರತಿ) transfer_ಸರತಿ->cmd_pool.destroy(device);

    // Explicitly clear to ensure queues drop their shared_ptrs to handles
    // before the Vulkan logical device instance is destroyed
    compute_ಸರತಿ.reset();
    transfer_ಸರತಿ.reset();

    ಗಾಗಿ (ಸ್ವಯಂಚಾಲಿತ& pipeline : all_pipelines) {
        ಆದರೆ (pipeline.expired()) {
            ಮುಂದುವರಿ;
        }

        vk_pipeline pl = pipeline.ಬೀಗ();
        ggml_vk_destroy_pipeline(device, pl);
    }
    all_pipelines.clear();

    device.destroy_Descriptor_ಗಣ_Layout(dsl);

    device.destroy();
}

ಶೂನ್ಯ vk_perf_logger::print_timings(ತಾರ್ಕಿಕ force) {
    ಆದರೆ (timings.ಖಾಲಿಗಳು()) {
        ಹಿಂತಿರುಗಿ;
    }
    print_ಎಣಿಕೆ++;
    ಆದರೆ ((print_ಎಣಿಕೆ % vk_perf_logger_frequency) != 0 && !force) {
        ಹಿಂತಿರುಗಿ;
    }
    print_ಎಣಿಕೆ = 0;
    uint64_t ಒಟ್ಟು_all_op_times = 0;
    std::cerr << "----------------\nVulkan Timings:" << std::ಸಾಲು_ಅಂತ್ಯ;
    ಗಾಗಿ (ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ & t : timings) {
        uint64_t ಒಟ್ಟು_op_times = 0;
        ಗಾಗಿ (ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ & ಸಮಯ : t.second) {
            ಒಟ್ಟು_op_times += ಸಮಯ;
        }
        std::cerr << t.ಮೊದಲ << ": " << t.second.size() << " x " << (ಒಟ್ಟು_op_times / t.second.size() / 1000.0)
                  << " us = " << (ಒಟ್ಟು_op_times / 1000.0) << " us";

        // If we have as many flops entries as timing entries for the op, then compute and log the flops/S.
        ಸ್ವಯಂಚಾಲಿತ it = flops.find(t.ಮೊದಲ);
        ಆದರೆ (it != flops.end() && (it->second).size() == t.second.size()) {
            uint64_t ಒಟ್ಟು_op_flops = 0;
            ಗಾಗಿ (ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ & elem : it->second) {
                ಒಟ್ಟು_op_flops += elem;
            }
            std::cerr << " ("
                      << (ದ್ವಿಗುಣ(ಒಟ್ಟು_op_flops) / (1000.0 * 1000.0 * 1000.0)) /
                             (ದ್ವಿಗುಣ(ಒಟ್ಟು_op_times) / (1000.0 * 1000.0 * 1000.0))
                      << " GFLOPS/s)";
        }

        ಒಟ್ಟು_all_op_times += ಒಟ್ಟು_op_times;

        std::cerr << std::ಸಾಲು_ಅಂತ್ಯ;
    }

    ಆದರೆ (timings.size() > 0) {
        std::cerr << "Total time: " << ಒಟ್ಟು_all_op_times / 1000.0 << " us." << std::ಸಾಲು_ಅಂತ್ಯ;
    }

    timings.clear();
    flops.clear();
}

std::string vk_perf_logger::get_ಗಂಟು_fusion_name(ಸ್ಥಿರಾಂಕ ggml_tensor * ಗಂಟು, ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *fusion_name, uint64_t *n_flops) {
    *n_flops = ggml_vk_get_ಗಂಟು_flops(ಗಂಟು);
    std::string fusion_str;
    ಆದರೆ (fusion_name) {
        fusion_str = fusion_name + std::string(" ");
    }
    ಆದರೆ (ಗಂಟು->op == GGML_OP_UNARY) {
        ಹಿಂತಿರುಗಿ fusion_str + ggml_unary_op_name(ggml_get_unary_op(ಗಂಟು));
    }
    ಆದರೆ (ಗಂಟು->op == GGML_OP_MUL_MAT || ಗಂಟು->op == GGML_OP_MUL_MAT_ID) {
        ಸ್ಥಿರಾಂಕ uint64_t m     = ಗಂಟು->ne[0];
        ಸ್ಥಿರಾಂಕ uint64_t n     = ಗಂಟು->ne[1];
        ಸ್ಥಿರಾಂಕ uint64_t k     = ಗಂಟು->src[1]->ne[0];
        ಸ್ಥಿರಾಂಕ uint64_t batch = ಗಂಟು->ne[2] * ಗಂಟು->ne[3];
        std::string    name  = ggml_op_name(ಗಂಟು->op);
        ಆದರೆ ((ಗಂಟು->op == GGML_OP_MUL_MAT && n <= mul_mat_vec_max_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ) ||
            (ಗಂಟು->op == GGML_OP_MUL_MAT_ID && ಗಂಟು->src[2]->ne[1] == 1)) {
            name += "_VEC";
        }
        name += " ";
        name += ggml_type_name(ಗಂಟು->src[0]->type);
        name += " m=" + std::to_string(m) + " n=" + std::to_string(n) + " k=" + std::to_string(k);
        ಆದರೆ (ಗಂಟು->op == GGML_OP_MUL_MAT_ID) {
            name += " n_expert=" + std::to_string(ಗಂಟು->src[0]->ne[2]);
        }
        ಆದರೆ (batch > 1) {
            name += " batch=" + std::to_string(batch);
        }
        ಹಿಂತಿರುಗಿ fusion_str + name;
    }
    ಆದರೆ (ಗಂಟು->op == GGML_OP_CONV_2D || ಗಂಟು->op == GGML_OP_CONV_TRANSPOSE_2D) {
        std::string   name    = ggml_op_name(ಗಂಟು->op);
        ಸ್ಥಿರಾಂಕ ggml_tensor * knl = ಗಂಟು->src[0];
        uint64_t      Cout    = ಗಂಟು->ne[2];
        uint64_t      ಗಾತ್ರ_K  = ಗಂಟು->src[1]->ne[2] * knl->ne[0] * knl->ne[1];
        uint64_t      ಗಾತ್ರ_N_2  = ಗಂಟು->ne[3] * ಗಂಟು->ne[0] * ಗಂಟು->ne[1];
        name += " M=Cout=" + std::to_string(Cout) + ", K=Cin*KW*KH=" + std::to_string(ಗಾತ್ರ_K) +
                ", N=N*OW*OH=" + std::to_string(ಗಾತ್ರ_N_2);
        ಹಿಂತಿರುಗಿ fusion_str + name;
    }
    ಆದರೆ (ಗಂಟು->op == GGML_OP_RMS_NORM) {
        std::string   name    = ggml_op_name(ಗಂಟು->op);
        name += "(" + std::to_string(ಗಂಟು->ne[0]) + "," + std::to_string(ಗಂಟು->ne[1]) + "," + std::to_string(ಗಂಟು->ne[2]) + "," + std::to_string(ಗಂಟು->ne[3]) + ")";
        ಹಿಂತಿರುಗಿ fusion_str + name;
    }
    ಆದರೆ (ಗಂಟು->op == GGML_OP_FLASH_ATTN_EXT) {
        ಸ್ಥಿರಾಂಕ ggml_tensor * dst = ಗಂಟು;
        ಸ್ಥಿರಾಂಕ ggml_tensor * q = ಗಂಟು->src[0];
        ಸ್ಥಿರಾಂಕ ggml_tensor * k = ಗಂಟು->src[1];
        ಸ್ಥಿರಾಂಕ ggml_tensor * v = ಗಂಟು->src[2];
        ಸ್ಥಿರಾಂಕ ggml_tensor * m = ಗಂಟು->src[3];
        std::stringstream name;
        name << fusion_str;
        name << ggml_op_name(ಗಂಟು->op) <<
            " dst(" << dst->ne[0] << "," << dst->ne[1] << "," << dst->ne[2] << "," << dst->ne[3] << "), " <<
            " q(" << q->ne[0] << "," << q->ne[1] << "," << q->ne[2] << "," << q->ne[3] << "), " <<
            " k(" << k->ne[0] << "," << k->ne[1] << "," << k->ne[2] << "," << k->ne[3] << "), " <<
            " v(" << v->ne[0] << "," << v->ne[1] << "," << v->ne[2] << "," << v->ne[3] << "), " <<
            " m(" << (m?m->ne[0]:0) << "," << (m?m->ne[1]:0) << "," << (m?m->ne[2]:0) << "," << (m?m->ne[3]:0) << ")";
        ಹಿಂತಿರುಗಿ name.str();
    }
    ಆದರೆ (ಗಂಟು->op == GGML_OP_ತುದಿ_K) {
        std::stringstream name;
        name << fusion_str;
        name << ggml_op_name(ಗಂಟು->op) <<
            " K=" << ಗಂಟು->ne[0] <<
            " (" << ಗಂಟು->src[0]->ne[0] << "," << ಗಂಟು->src[0]->ne[1] << "," << ಗಂಟು->src[0]->ne[2] << "," << ಗಂಟು->src[0]->ne[3] << ")";
        ಹಿಂತಿರುಗಿ name.str();
    }
    ಹಿಂತಿರುಗಿ fusion_str + ggml_op_name(ಗಂಟು->op);
}

ggml_backend_vk_ಬಫರ್_context::~ggml_backend_vk_ಬಫರ್_context() {
    ggml_vk_destroy_ಬಫರ್(dev_ಬಫರ್);
}

ggml_vk_debug_label::ggml_vk_debug_label(vk_context & ctx, ಸ್ಥಿರಾಂಕ std::string & pipeline_name, uint32_t wg0, uint32_t wg1, uint32_t wg2) {
    ಆದರೆ (!vk_instance.debug_utils_support || ctx->s == ಶೂನ್ಯಸೂಚಕ) {
        ಹಿಂತಿರುಗಿ;
    }
    begin(ctx, pipeline_name + " (" + std::to_string(wg0) + "," + std::to_string(wg1) + "," + std::to_string(wg2) + ")");
}

ggml_vk_debug_label::ggml_vk_debug_label(vk_context & ctx, ಸ್ಥಿರಾಂಕ ggml_cgraph * cgraph, ಪೂರ್ಣಾಂಕ ಗಂಟು_idx, ಪೂರ್ಣಾಂಕ n_fused) {
    ಆದರೆ (!vk_instance.debug_utils_support || ctx->s == ಶೂನ್ಯಸೂಚಕ) {
        ಹಿಂತಿರುಗಿ;
    }
    std::string name = ggml_op_name(cgraph->nodes[ಗಂಟು_idx]->op);
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 1; i <= n_fused; i++) {
        name += "+";
        name += ggml_op_name(cgraph->nodes[ಗಂಟು_idx + i]->op);
    }
    name += " ";
    name += cgraph->nodes[ಗಂಟು_idx]->name;
    begin(ctx, name);
}

ggml_vk_debug_label::ggml_vk_debug_label(vk_ಸರತಿ_handle * handle, ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * name) {
    ಆದರೆ (!vk_instance.debug_utils_support || handle == ಶೂನ್ಯಸೂಚಕ) {
        ಹಿಂತಿರುಗಿ;
    }
    vk::DebugUtilsLabelEXT label = {};
    label.pLabelName = name;
    label.color = std::array<ದಶಮಾಂಶ, 4>{1.0f, 1.0f, 1.0f, 1.0f};

    qhandle = handle;
    std::lock_guard<vk_ಸರತಿ_handle> guard(*qhandle);
    vk_instance.pfn_vk_ಸರತಿ_Begin_Debug_Utils_Label_EXT(qhandle->queue, reinterpret_cast<VkDebugUtilsLabelEXT *>(&label));
}

ಶೂನ್ಯ ggml_vk_debug_label::close() {
    ಆದರೆ (subctx != ಶೂನ್ಯಸೂಚಕ) {
        // close on the current command buffer, which may differ from the one begin used
        ಆದರೆ (subctx->s != ಶೂನ್ಯಸೂಚಕ) {
            vk_instance.pfn_vk_Cmd_ಕೊನೆ_Debug_Utils_Label_EXT(subctx->s->ಬಫರ್->buf);
        }
        subctx->debug_labels.pop_back();
        subctx = ಶೂನ್ಯಸೂಚಕ;
    }
    ಆದರೆ (qhandle != ಶೂನ್ಯಸೂಚಕ) {
        std::lock_guard<vk_ಸರತಿ_handle> guard(*qhandle);
        vk_instance.pfn_vk_ಸರತಿ_ಕೊನೆ_Debug_Utils_Label_EXT(qhandle->queue);
        qhandle = ಶೂನ್ಯಸೂಚಕ;
    }
}

ಶೂನ್ಯ ggml_vk_debug_label::begin(vk_context & ctx, ಸ್ಥಿರಾಂಕ std::string & name) {
    ಆದರೆ (!vk_instance.debug_utils_support || ctx->s == ಶೂನ್ಯಸೂಚಕ) {
        ಹಿಂತಿರುಗಿ;
    }
    subctx = ctx.get();
    subctx->debug_labels.push_back(name);
    ggml_vk_cmd_label_begin(subctx->s->ಬಫರ್->buf, subctx->debug_labels.back().c_str());
}

