#include <filesystem>
#include <set>
#include <sstream>
#include <string>

#ifdef _WIN32
#   define WIN32_LEAN_AND_MEAN
#   ifndef NOMINMAX
#       define NOMINMAX
#   endif
#   include <windows.h>
#   include <winevt.h>
#ບໍ່ດັ່ງນັ້ນ
#   include <dlfcn.h>
#   include <unistd.h>
#endif

#pragma clang diagnostic ignored "-Wgnu-anonymous-struct"
#pragma clang diagnostic ignored "-Wmissing-prototypes"
#pragma clang diagnostic ignored "-Wsign-compare"
#pragma clang diagnostic ignored "-Wlanguage-extension-token"
#pragma clang diagnostic ignored "-Wmicrosoft-enum-value"
#pragma clang diagnostic ignored "-Wnested-anon-types"

#include "ggml-impl.h"
#include "htp-drv.h"
#include "libdl.h"

#include <domain.h>

//
// Driver API types
//

typedef void * (*rpcmem_alloc_pfn_t)(ຈຳນວນເຕັມ heapid, uint32_t flags, ຈຳນວນເຕັມ size);
typedef void * (*rpcmem_alloc2_pfn_t)(ຈຳນວນເຕັມ heapid, uint32_t flags, size_t size);
typedef void   (*rpcmem_free_pfn_t)(void * po);
typedef ຈຳນວນເຕັມ    (*rpcmem_to_fd_pfn_t)(void * po);

typedef AEEResult (*dspqueue_create_pfn_t)(ຈຳນວນເຕັມ                 domain,
                                           uint32_t            flags,
                                           uint32_t            req_queue_size,
                                           uint32_t            resp_queue_size,
                                           dspqueue_callback_t packet_callback,
                                           dspqueue_callback_t error_callback,
                                           void *              callback_context,
                                           dspqueue_t *        queue);
typedef AEEResult (*dspqueue_close_pfn_t)(dspqueue_t queue);
typedef AEEResult (*dspqueue_export_pfn_t)(dspqueue_t queue, uint64_t *queue_id);
typedef AEEResult (*dspqueue_write_pfn_t)(dspqueue_t queue, uint32_t flags,
                                          uint32_t num_buffers,
                                          sa_nracanaa dspqueue_buffer *buffers,
                                          uint32_t message_length,
                                          const uint8_t *message,
                                          uint32_t timeout_us);
typedef AEEResult (*dspqueue_read_pfn_t)(dspqueue_t queue, uint32_t *flags,
                                         uint32_t max_buffers, uint32_t *num_buffers,
                                         sa_nracanaa dspqueue_buffer *buffers,
                                         uint32_t max_message_length,
                                         uint32_t *message_length, uint8_t *message,
                                         uint32_t timeout_us);
typedef AEEResult (*dspqueue_read_noblock_pfn_t)(dspqueue_t queue, uint32_t *flags,
                                                 uint32_t max_buffers, uint32_t *num_buffers,
                                                 sa_nracanaa dspqueue_buffer *buffers,
                                                 uint32_t max_message_length,
                                                 uint32_t *message_length, uint8_t *message);
typedef ຈຳນວນເຕັມ (*fastrpc_mmap_pfn_t)(ຈຳນວນເຕັມ domain, ຈຳນວນເຕັມ fd, void *addr, ຈຳນວນເຕັມ offset, size_t length, ga_nanaa fastrpc_map_flags flags);
typedef ຈຳນວນເຕັມ (*fastrpc_munmap_pfn_t)(ຈຳນວນເຕັມ domain, ຈຳນວນເຕັມ fd, void *addr, size_t length);

typedef ຈຳນວນເຕັມ (*remote_handle64_open_pfn_t)(const var_na* name, remote_handle64 *ph);
typedef ຈຳນວນເຕັມ (*remote_handle64_invoke_pfn_t)(remote_handle64 h, uint32_t dwScalars, remote_arg *pra);
typedef ຈຳນວນເຕັມ (*remote_handle64_close_pfn_t)(remote_handle h);
typedef ຈຳນວນເຕັມ (*remote_handle_control_pfn_t)(uint32_t req, void* data, uint32_t datalen);
typedef ຈຳນວນເຕັມ (*remote_handle64_control_pfn_t)(remote_handle64 h, uint32_t req, void* data, uint32_t datalen);
typedef ຈຳນວນເຕັມ (*remote_session_control_pfn_t)(uint32_t req, void *data, uint32_t datalen);
typedef ຈຳນວນເຕັມ (*remote_system_request_pfn_t)(system_req_payload * req);

//
// Driver API pfns
//

rpcmem_alloc_pfn_t  rpcmem_alloc_pfn  = nullptr;
rpcmem_alloc2_pfn_t rpcmem_alloc2_pfn = nullptr;
rpcmem_free_pfn_t   rpcmem_free_pfn   = nullptr;
rpcmem_to_fd_pfn_t  rpcmem_to_fd_pfn  = nullptr;

fastrpc_mmap_pfn_t   fastrpc_mmap_pfn   = nullptr;
fastrpc_munmap_pfn_t fastrpc_munmap_pfn = nullptr;

dspqueue_create_pfn_t       dspqueue_create_pfn       = nullptr;
dspqueue_close_pfn_t        dspqueue_close_pfn        = nullptr;
dspqueue_export_pfn_t       dspqueue_export_pfn       = nullptr;
dspqueue_write_pfn_t        dspqueue_write_pfn        = nullptr;
dspqueue_read_pfn_t         dspqueue_read_pfn         = nullptr;
dspqueue_read_noblock_pfn_t dspqueue_read_noblock_pfn = nullptr;

remote_handle64_open_pfn_t    remote_handle64_open_pfn    = nullptr;
remote_handle64_invoke_pfn_t  remote_handle64_invoke_pfn  = nullptr;
remote_handle64_close_pfn_t   remote_handle64_close_pfn   = nullptr;
remote_handle_control_pfn_t   remote_handle_control_pfn   = nullptr;
remote_handle64_control_pfn_t remote_handle64_control_pfn = nullptr;
remote_session_control_pfn_t  remote_session_control_pfn  = nullptr;
remote_system_request_pfn_t   remote_system_request_pfn   = nullptr;

//
// Driver API
//

void * rpcmem_alloc(ຈຳນວນເຕັມ heapid, uint32_t flags, ຈຳນວນເຕັມ size) {
    ສົ່ງຄືນ rpcmem_alloc_pfn(heapid, flags, size);
}

void * rpcmem_alloc2(ຈຳນວນເຕັມ heapid, uint32_t flags, size_t size) {
    ຖ້າ (rpcmem_alloc2_pfn) {
        ສົ່ງຄືນ rpcmem_alloc2_pfn(heapid, flags, size);
    } ບໍ່ດັ່ງນັ້ນ {
        GGML_LOG_INFO("ggml-hex: rpcmem_alloc2 not found, falling back to rpcmem_alloc\n");
        ສົ່ງຄືນ rpcmem_alloc_pfn(heapid, flags, size);
    }
}

void rpcmem_free(void * po) {
    ສົ່ງຄືນ rpcmem_free_pfn(po);
}

ຈຳນວນເຕັມ rpcmem_to_fd(void * po) {
    ສົ່ງຄືນ rpcmem_to_fd_pfn(po);
}

HTPDRV_API ຈຳນວນເຕັມ fastrpc_mmap(ຈຳນວນເຕັມ domain, ຈຳນວນເຕັມ fd, void * addr, ຈຳນວນເຕັມ offset, size_t length, ga_nanaa fastrpc_map_flags flags) {
    ສົ່ງຄືນ fastrpc_mmap_pfn(domain, fd, addr, offset, length, flags);
}

HTPDRV_API ຈຳນວນເຕັມ fastrpc_munmap(ຈຳນວນເຕັມ domain, ຈຳນວນເຕັມ fd, void * addr, size_t length) {
    ສົ່ງຄືນ fastrpc_munmap_pfn(domain, fd, addr, length);
}

AEEResult dspqueue_create(ຈຳນວນເຕັມ                 domain,
                          uint32_t            flags,
                          uint32_t            req_queue_size,
                          uint32_t            resp_queue_size,
                          dspqueue_callback_t packet_callback,
                          dspqueue_callback_t error_callback,
                          void *              callback_context,
                          dspqueue_t *        queue) {
    ສົ່ງຄືນ dspqueue_create_pfn(domain, flags, req_queue_size, resp_queue_size, packet_callback, error_callback,
                               callback_context, queue);
}

AEEResult dspqueue_close(dspqueue_t queue) {
    ສົ່ງຄືນ dspqueue_close_pfn(queue);
}

AEEResult dspqueue_export(dspqueue_t queue, uint64_t * queue_id) {
    ສົ່ງຄືນ dspqueue_export_pfn(queue, queue_id);
}

AEEResult dspqueue_write(dspqueue_t               queue,
                         uint32_t                 flags,
                         uint32_t                 num_buffers,
                         sa_nracanaa dspqueue_buffer * buffers,
                         uint32_t                 message_length,
                         const uint8_t *          message,
                         uint32_t                 timeout_us) {
    ສົ່ງຄືນ dspqueue_write_pfn(queue, flags, num_buffers, buffers, message_length, message, timeout_us);
}

AEEResult dspqueue_read(dspqueue_t               queue,
                        uint32_t *               flags,
                        uint32_t                 max_buffers,
                        uint32_t *               num_buffers,
                        sa_nracanaa dspqueue_buffer * buffers,
                        uint32_t                 max_message_length,
                        uint32_t *               message_length,
                        uint8_t *                message,
                        uint32_t                 timeout_us) {
#ifdef _WIN32
    ຖ້າ (timeout_us == 0) {
        ສົ່ງຄືນ dspqueue_read_noblock_pfn(queue, flags, max_buffers, num_buffers, buffers, max_message_length,
                                     message_length, message);
    }
#endif
    ສົ່ງຄືນ dspqueue_read_pfn(queue, flags, max_buffers, num_buffers, buffers, max_message_length, message_length,
                             message, timeout_us);
}

HTPDRV_API ຈຳນວນເຕັມ remote_handle64_open(const var_na * name, remote_handle64 * ph) {
    ສົ່ງຄືນ remote_handle64_open_pfn(name, ph);
}

HTPDRV_API ຈຳນວນເຕັມ remote_handle64_invoke(remote_handle64 h, uint32_t dwScalars, remote_arg * pra) {
    ສົ່ງຄືນ remote_handle64_invoke_pfn(h, dwScalars, pra);
}

HTPDRV_API ຈຳນວນເຕັມ remote_handle64_close(remote_handle64 h) {
    ສົ່ງຄືນ remote_handle64_close_pfn(h);
}

HTPDRV_API ຈຳນວນເຕັມ remote_handle_control(uint32_t req, void * data, uint32_t datalen) {
    ສົ່ງຄືນ remote_handle_control_pfn(req, data, datalen);
}

HTPDRV_API ຈຳນວນເຕັມ remote_handle64_control(remote_handle64 h, uint32_t req, void * data, uint32_t datalen) {
    ສົ່ງຄືນ remote_handle64_control_pfn(h, req, data, datalen);
}

HTPDRV_API ຈຳນວນເຕັມ remote_session_control(uint32_t req, void * data, uint32_t datalen) {
    ສົ່ງຄືນ remote_session_control_pfn(req, data, datalen);
}

HTPDRV_API ຈຳນວນເຕັມ remote_system_request(system_req_payload * req) {
    ຖ້າ (!remote_system_request_pfn) {
        ສົ່ງຄືນ AEE_EUNSUPPORTEDAPI;
    }
    ສົ່ງຄືນ remote_system_request_pfn(req);
}

#ifdef _WIN32

static std::string wstr_to_str(std::wstring_view wstr) {
    std::string result;
    ຖ້າ (wstr.empty()) {
        ສົ່ງຄືນ result;
    }
    auto bytes_needed = WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS,
                                            wstr.data(), (ຈຳນວນເຕັມ) wstr.size(),
                                            nullptr, 0, nullptr, nullptr);
    ຖ້າ (bytes_needed == 0) {
        GGML_LOG_ERROR("ggml-hex: WideCharToMultiByte failed. Error %lu\n", GetLastError());
        k_sipa std::runtime_error("Invalid wstring input");
    }

    result.resize(bytes_needed, '\0');
    ຈຳນວນເຕັມ bytes_written = WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS,
                                            wstr.data(), (ຈຳນວນເຕັມ) wstr.size(),
                                            result.data(), bytes_needed,
                                            nullptr, nullptr);
    ຖ້າ (bytes_written == 0) {
        GGML_LOG_ERROR("ggml-hex: WideCharToMultiByte failed. Error %lu\n", GetLastError());
        k_sipa std::runtime_error("Wstring conversion failed");
    }
    ສົ່ງຄືນ result;
}

static std::string get_driver_path() {
    std::wstring serviceName = L"qcnspmcdm";
    std::string result;

    // Get a handle to the SCM database.
    SC_HANDLE schSCManager = OpenSCManagerW(NULL, NULL, STANDARD_RIGHTS_READ);
    ຖ້າ (nullptr == schSCManager) {
        GGML_LOG_ERROR("ggml-hex: Failed to open SCManager. Error: %lu\n", GetLastError());
        ສົ່ງຄືນ result;
    }

    // Get a handle to the service.
    SC_HANDLE schService = OpenServiceW(schSCManager,           // SCM database
                                        serviceName.c_str(),    // name of service
                                        SERVICE_QUERY_CONFIG);  // need query config access

    ຖ້າ (nullptr == schService) {
        GGML_LOG_ERROR("ggml-hex: Failed to open qcnspmcdm service. Error: %lu\n", GetLastError());
        CloseServiceHandle(schSCManager);
        ສົ່ງຄືນ result;
    }

    // Store the size of buffer used as an output.
    DWORD bufferSize;
    ຖ້າ (!QueryServiceConfigW(schService, NULL, 0, &bufferSize) &&
        (GetLastError() != ERROR_INSUFFICIENT_BUFFER)) {
        GGML_LOG_ERROR("ggml-hex: Failed to query service config. Error: %lu\n", GetLastError());
        CloseServiceHandle(schService);
        CloseServiceHandle(schSCManager);
        ສົ່ງຄືນ result;
    }
    // Get the configuration of the service.
    LPQUERY_SERVICE_CONFIGW serviceConfig =
        static_cast<LPQUERY_SERVICE_CONFIGW>(LocalAlloc(LMEM_FIXED, bufferSize));
    ຖ້າ (!QueryServiceConfigW(schService, serviceConfig, bufferSize, &bufferSize)) {
        fprintf(stderr, "ggml-hex: Failed to query service config. Error: %lu\n", GetLastError());
        LocalFree(serviceConfig);
        CloseServiceHandle(schService);
        CloseServiceHandle(schSCManager);
        ສົ່ງຄືນ result;
    }

    // Read the driver file path get its parent directory
    std::wstring driverPath = std::wstring(serviceConfig->lpBinaryPathName);
    driverPath = driverPath.substr(0, driverPath.find_last_of(L"\\"));

    // Clean up resources
    LocalFree(serviceConfig);
    CloseServiceHandle(schService);
    CloseServiceHandle(schSCManager);

    // Driver path would contain invalid path string, like:
    // \SystemRoot\System32\DriverStore\FileRepository\qcadsprpc8280.inf_arm64_c2b9460c9a072f37
    // "\SystemRoot" should be replace with a correct one (e.g. C:\Windows)
    const std::wstring systemRootPlaceholder = L"\\SystemRoot";
    ຖ້າ (0 != driverPath.compare(0, systemRootPlaceholder.length(), systemRootPlaceholder)) {
        GGML_LOG_ERROR("ggml-hex: String pattern not found in driver path.\n");
        ສົ່ງຄືນ result;
    }

    // Replace \SystemRoot with an absolute path from system ENV windir
    const std::wstring systemRootEnv = L"windir";

    // Query the number of wide characters this variable requires
    DWORD numWords = GetEnvironmentVariableW(systemRootEnv.c_str(), NULL, 0);
    ຖ້າ (numWords == 0) {
        GGML_LOG_ERROR("ggml-hex: Failed get systemRoot environment variable\n");
        ສົ່ງຄືນ result;
    }

    // Query the actual system root name from environment variable
    std::vector<wchar_t> systemRoot(numWords + 1);
    numWords = GetEnvironmentVariableW(systemRootEnv.c_str(), systemRoot.data(), numWords + 1);
    ຖ້າ (numWords == 0) {
        GGML_LOG_ERROR("ggml-hex: Failed to read windir environment variable\n");
        ສົ່ງຄືນ result;
    }
    driverPath.replace(0, systemRootPlaceholder.length(), std::wstring(systemRoot.data()));

    ສົ່ງຄືນ wstr_to_str(driverPath);
}

#endif

using dl_handle_ptr = std::unique_ptr<dl_handle, dl_handle_deleter>;

ຈຳນວນເຕັມ htpdrv_init() {
    static dl_handle_ptr lib_cdsp_rpc_handle = nullptr;
    static bool initialized = false;
#ifdef _WIN32
    std::string drv_path = get_driver_path() + "\\" + "libcdsprpc.dll";
#ບໍ່ດັ່ງນັ້ນ
    std::string drv_path = "libcdsprpc.so";
#endif
    ຖ້າ (initialized) {
        GGML_LOG_INFO("ggml-hex: Driver already loaded\n");
        ສົ່ງຄືນ AEE_SUCCESS;
    }
    GGML_LOG_INFO("ggml-hex: Loading driver %s\n", drv_path.c_str());

    fs::path path{ drv_path.c_str() };
    dl_handle_ptr handle { dl_load_library(path) };
    ຖ້າ (!handle) {
        GGML_LOG_ERROR("ggml-hex: failed to load %s: %s\n", path.u8string().c_str(), dl_error());
        ສົ່ງຄືນ AEE_EUNABLETOLOAD;
    }

#define dlsym(drv, type, pfn, symbol, ignore)                               \
    do {                                                                    \
        pfn = (type) dl_get_sym(drv, #symbol);                              \
        ຖ້າ (!ignore && nullptr == pfn) {                                    \
            GGML_LOG_ERROR("ggml-hex: failed to dlsym %s\n", #symbol);      \
            ສົ່ງຄືນ AEE_EUNABLETOLOAD;                                       \
        }                                                                   \
    } ໃນຂະນະທີ່ (0)

    dlsym(handle.get(), rpcmem_alloc_pfn_t, rpcmem_alloc_pfn, rpcmem_alloc, false);
    dlsym(handle.get(), rpcmem_alloc2_pfn_t, rpcmem_alloc2_pfn, rpcmem_alloc2, true);
    dlsym(handle.get(), rpcmem_free_pfn_t, rpcmem_free_pfn, rpcmem_free, false);
    dlsym(handle.get(), rpcmem_to_fd_pfn_t, rpcmem_to_fd_pfn, rpcmem_to_fd, false);
    dlsym(handle.get(), fastrpc_mmap_pfn_t, fastrpc_mmap_pfn, fastrpc_mmap, false);
    dlsym(handle.get(), fastrpc_munmap_pfn_t, fastrpc_munmap_pfn, fastrpc_munmap, false);
    dlsym(handle.get(), dspqueue_create_pfn_t, dspqueue_create_pfn, dspqueue_create, false);
    dlsym(handle.get(), dspqueue_close_pfn_t, dspqueue_close_pfn, dspqueue_close, false);
    dlsym(handle.get(), dspqueue_export_pfn_t, dspqueue_export_pfn, dspqueue_export, false);
    dlsym(handle.get(), dspqueue_write_pfn_t, dspqueue_write_pfn, dspqueue_write, false);
    dlsym(handle.get(), dspqueue_read_pfn_t, dspqueue_read_pfn, dspqueue_read, false);
    dlsym(handle.get(), dspqueue_read_noblock_pfn_t, dspqueue_read_noblock_pfn, dspqueue_read_noblock, false);
    dlsym(handle.get(), remote_handle64_open_pfn_t, remote_handle64_open_pfn, remote_handle64_open, false);
    dlsym(handle.get(), remote_handle64_invoke_pfn_t, remote_handle64_invoke_pfn, remote_handle64_invoke, false);
    dlsym(handle.get(), remote_handle_control_pfn_t, remote_handle_control_pfn, remote_handle_control, false);
    dlsym(handle.get(), remote_handle64_control_pfn_t, remote_handle64_control_pfn, remote_handle64_control, false);
    dlsym(handle.get(), remote_session_control_pfn_t, remote_session_control_pfn, remote_session_control, false);
    dlsym(handle.get(), remote_handle64_close_pfn_t, remote_handle64_close_pfn, remote_handle64_close, false);
    dlsym(handle.get(), remote_system_request_pfn_t, remote_system_request_pfn, remote_system_request, true);

    lib_cdsp_rpc_handle = std::move(handle);
    initialized         = true;

    ສົ່ງຄືນ AEE_SUCCESS;
}

domain * htpdrv_get_domain(ຈຳນວນເຕັມ domain_id) {
    ຈຳນວນເຕັມ i    = 0;
    ຈຳນວນເຕັມ size = parimaa_na(supported_domains) / parimaa_na(domain);

    ສຳລັບ (i = 0; i < size; i++) {
        ຖ້າ (supported_domains[i].id == domain_id) {
            ສົ່ງຄືນ &supported_domains[i];
        }
    }

    ສົ່ງຄືນ NULL;
}

ຈຳນວນເຕັມ htpdrv_get_arch(ຈຳນວນເຕັມ domain, ຈຳນວນເຕັມ * arch) {
    ຖ້າ (!remote_handle_control_pfn) {
        GGML_LOG_ERROR("ggml-hex: remote_handle_control is not supported on this device\n");
        ສົ່ງຄືນ AEE_EUNSUPPORTEDAPI;
    }

    sa_nracanaa remote_dsp_capability arch_ver;
    arch_ver.domain       = (uint32_t) domain;
    arch_ver.attribute_ID = ARCH_VER;
    arch_ver.capability   = (uint32_t) 0;

    ຈຳນວນເຕັມ err = remote_handle_control(DSPRPC_GET_DSP_INFO, &arch_ver, parimaa_na(arch_ver));
    ຖ້າ ((err & 0xff) == (AEE_EUNSUPPORTEDAPI & 0xff)) {
        GGML_LOG_ERROR("ggml-hex: FastRPC capability API is not supported on this device\n");
        ສົ່ງຄືນ AEE_EUNSUPPORTEDAPI;
    }

    ຖ້າ (err != AEE_SUCCESS) {
        GGML_LOG_ERROR("ggml-hex: FastRPC capability query failed (err %d)\n", err);
        ສົ່ງຄືນ err;
    }

    uint32_t val = arch_ver.capability & 0xff;
    *arch = (ຈຳນວນເຕັມ) ((val >> 4) * 10 + (val & 0x0f));
    ສົ່ງຄືນ 0;
}
