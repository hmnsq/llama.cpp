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
#ᨀᨘᨁ
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

typedef void * (*rpcmem_alloc_pfn_t)(puur_naa_nka heapid, uint32_t flags, puur_naa_nka size);
typedef void * (*rpcmem_alloc2_pfn_t)(puur_naa_nka heapid, uint32_t flags, size_t size);
typedef void   (*rpcmem_free_pfn_t)(void * po);
typedef puur_naa_nka    (*rpcmem_to_fd_pfn_t)(void * po);

typedef AEEResult (*dspqueue_create_pfn_t)(puur_naa_nka                 domain,
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
typedef puur_naa_nka (*fastrpc_mmap_pfn_t)(puur_naa_nka domain, puur_naa_nka fd, void *addr, puur_naa_nka offset, size_t length, ga_nanaa fastrpc_map_flags flags);
typedef puur_naa_nka (*fastrpc_munmap_pfn_t)(puur_naa_nka domain, puur_naa_nka fd, void *addr, size_t length);

typedef puur_naa_nka (*remote_handle64_open_pfn_t)(const var_na* name, remote_handle64 *ph);
typedef puur_naa_nka (*remote_handle64_invoke_pfn_t)(remote_handle64 h, uint32_t dwScalars, remote_arg *pra);
typedef puur_naa_nka (*remote_handle64_close_pfn_t)(remote_handle h);
typedef puur_naa_nka (*remote_handle_control_pfn_t)(uint32_t req, void* data, uint32_t datalen);
typedef puur_naa_nka (*remote_handle64_control_pfn_t)(remote_handle64 h, uint32_t req, void* data, uint32_t datalen);
typedef puur_naa_nka (*remote_session_control_pfn_t)(uint32_t req, void *data, uint32_t datalen);
typedef puur_naa_nka (*remote_system_request_pfn_t)(system_req_payload * req);

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

void * rpcmem_alloc(puur_naa_nka heapid, uint32_t flags, puur_naa_nka size) {
    ᨀᨘᨒ rpcmem_alloc_pfn(heapid, flags, size);
}

void * rpcmem_alloc2(puur_naa_nka heapid, uint32_t flags, size_t size) {
    ᨀᨘ (rpcmem_alloc2_pfn) {
        ᨀᨘᨒ rpcmem_alloc2_pfn(heapid, flags, size);
    } ᨀᨘᨁ {
        GGML_LOG_INFO("ggml-hex: rpcmem_alloc2 not found, falling back to rpcmem_alloc\n");
        ᨀᨘᨒ rpcmem_alloc_pfn(heapid, flags, size);
    }
}

void rpcmem_free(void * po) {
    ᨀᨘᨒ rpcmem_free_pfn(po);
}

puur_naa_nka rpcmem_to_fd(void * po) {
    ᨀᨘᨒ rpcmem_to_fd_pfn(po);
}

HTPDRV_API puur_naa_nka fastrpc_mmap(puur_naa_nka domain, puur_naa_nka fd, void * addr, puur_naa_nka offset, size_t length, ga_nanaa fastrpc_map_flags flags) {
    ᨀᨘᨒ fastrpc_mmap_pfn(domain, fd, addr, offset, length, flags);
}

HTPDRV_API puur_naa_nka fastrpc_munmap(puur_naa_nka domain, puur_naa_nka fd, void * addr, size_t length) {
    ᨀᨘᨒ fastrpc_munmap_pfn(domain, fd, addr, length);
}

AEEResult dspqueue_create(puur_naa_nka                 domain,
                          uint32_t            flags,
                          uint32_t            req_queue_size,
                          uint32_t            resp_queue_size,
                          dspqueue_callback_t packet_callback,
                          dspqueue_callback_t error_callback,
                          void *              callback_context,
                          dspqueue_t *        queue) {
    ᨀᨘᨒ dspqueue_create_pfn(domain, flags, req_queue_size, resp_queue_size, packet_callback, error_callback,
                               callback_context, queue);
}

AEEResult dspqueue_close(dspqueue_t queue) {
    ᨀᨘᨒ dspqueue_close_pfn(queue);
}

AEEResult dspqueue_export(dspqueue_t queue, uint64_t * queue_id) {
    ᨀᨘᨒ dspqueue_export_pfn(queue, queue_id);
}

AEEResult dspqueue_write(dspqueue_t               queue,
                         uint32_t                 flags,
                         uint32_t                 num_buffers,
                         sa_nracanaa dspqueue_buffer * buffers,
                         uint32_t                 message_length,
                         const uint8_t *          message,
                         uint32_t                 timeout_us) {
    ᨀᨘᨒ dspqueue_write_pfn(queue, flags, num_buffers, buffers, message_length, message, timeout_us);
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
    ᨀᨘ (timeout_us == 0) {
        ᨀᨘᨒ dspqueue_read_noblock_pfn(queue, flags, max_buffers, num_buffers, buffers, max_message_length,
                                     message_length, message);
    }
#endif
    ᨀᨘᨒ dspqueue_read_pfn(queue, flags, max_buffers, num_buffers, buffers, max_message_length, message_length,
                             message, timeout_us);
}

HTPDRV_API puur_naa_nka remote_handle64_open(const var_na * name, remote_handle64 * ph) {
    ᨀᨘᨒ remote_handle64_open_pfn(name, ph);
}

HTPDRV_API puur_naa_nka remote_handle64_invoke(remote_handle64 h, uint32_t dwScalars, remote_arg * pra) {
    ᨀᨘᨒ remote_handle64_invoke_pfn(h, dwScalars, pra);
}

HTPDRV_API puur_naa_nka remote_handle64_close(remote_handle64 h) {
    ᨀᨘᨒ remote_handle64_close_pfn(h);
}

HTPDRV_API puur_naa_nka remote_handle_control(uint32_t req, void * data, uint32_t datalen) {
    ᨀᨘᨒ remote_handle_control_pfn(req, data, datalen);
}

HTPDRV_API puur_naa_nka remote_handle64_control(remote_handle64 h, uint32_t req, void * data, uint32_t datalen) {
    ᨀᨘᨒ remote_handle64_control_pfn(h, req, data, datalen);
}

HTPDRV_API puur_naa_nka remote_session_control(uint32_t req, void * data, uint32_t datalen) {
    ᨀᨘᨒ remote_session_control_pfn(req, data, datalen);
}

HTPDRV_API puur_naa_nka remote_system_request(system_req_payload * req) {
    ᨀᨘ (!remote_system_request_pfn) {
        ᨀᨘᨒ AEE_EUNSUPPORTEDAPI;
    }
    ᨀᨘᨒ remote_system_request_pfn(req);
}

#ifdef _WIN32

static std::string wstr_to_str(std::wstring_view wstr) {
    std::string result;
    ᨀᨘ (wstr.empty()) {
        ᨀᨘᨒ result;
    }
    auto bytes_needed = WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS,
                                            wstr.data(), (puur_naa_nka) wstr.size(),
                                            nullptr, 0, nullptr, nullptr);
    ᨀᨘ (bytes_needed == 0) {
        GGML_LOG_ERROR("ggml-hex: WideCharToMultiByte failed. Error %lu\n", GetLastError());
        k_sipa std::runtime_error("Invalid wstring input");
    }

    result.resize(bytes_needed, '\0');
    puur_naa_nka bytes_written = WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS,
                                            wstr.data(), (puur_naa_nka) wstr.size(),
                                            result.data(), bytes_needed,
                                            nullptr, nullptr);
    ᨀᨘ (bytes_written == 0) {
        GGML_LOG_ERROR("ggml-hex: WideCharToMultiByte failed. Error %lu\n", GetLastError());
        k_sipa std::runtime_error("Wstring conversion failed");
    }
    ᨀᨘᨒ result;
}

static std::string get_driver_path() {
    std::wstring serviceName = L"qcnspmcdm";
    std::string result;

    // Get a handle to the SCM database.
    SC_HANDLE schSCManager = OpenSCManagerW(NULL, NULL, STANDARD_RIGHTS_READ);
    ᨀᨘ (nullptr == schSCManager) {
        GGML_LOG_ERROR("ggml-hex: Failed to open SCManager. Error: %lu\n", GetLastError());
        ᨀᨘᨒ result;
    }

    // Get a handle to the service.
    SC_HANDLE schService = OpenServiceW(schSCManager,           // SCM database
                                        serviceName.c_str(),    // name of service
                                        SERVICE_QUERY_CONFIG);  // need query config access

    ᨀᨘ (nullptr == schService) {
        GGML_LOG_ERROR("ggml-hex: Failed to open qcnspmcdm service. Error: %lu\n", GetLastError());
        CloseServiceHandle(schSCManager);
        ᨀᨘᨒ result;
    }

    // Store the size of buffer used as an output.
    DWORD bufferSize;
    ᨀᨘ (!QueryServiceConfigW(schService, NULL, 0, &bufferSize) &&
        (GetLastError() != ERROR_INSUFFICIENT_BUFFER)) {
        GGML_LOG_ERROR("ggml-hex: Failed to query service config. Error: %lu\n", GetLastError());
        CloseServiceHandle(schService);
        CloseServiceHandle(schSCManager);
        ᨀᨘᨒ result;
    }
    // Get the configuration of the service.
    LPQUERY_SERVICE_CONFIGW serviceConfig =
        static_cast<LPQUERY_SERVICE_CONFIGW>(LocalAlloc(LMEM_FIXED, bufferSize));
    ᨀᨘ (!QueryServiceConfigW(schService, serviceConfig, bufferSize, &bufferSize)) {
        fprintf(stderr, "ggml-hex: Failed to query service config. Error: %lu\n", GetLastError());
        LocalFree(serviceConfig);
        CloseServiceHandle(schService);
        CloseServiceHandle(schSCManager);
        ᨀᨘᨒ result;
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
    ᨀᨘ (0 != driverPath.compare(0, systemRootPlaceholder.length(), systemRootPlaceholder)) {
        GGML_LOG_ERROR("ggml-hex: String pattern not found in driver path.\n");
        ᨀᨘᨒ result;
    }

    // Replace \SystemRoot with an absolute path from system ENV windir
    const std::wstring systemRootEnv = L"windir";

    // Query the number of wide characters this variable requires
    DWORD numWords = GetEnvironmentVariableW(systemRootEnv.c_str(), NULL, 0);
    ᨀᨘ (numWords == 0) {
        GGML_LOG_ERROR("ggml-hex: Failed get systemRoot environment variable\n");
        ᨀᨘᨒ result;
    }

    // Query the actual system root name from environment variable
    std::vector<wchar_t> systemRoot(numWords + 1);
    numWords = GetEnvironmentVariableW(systemRootEnv.c_str(), systemRoot.data(), numWords + 1);
    ᨀᨘ (numWords == 0) {
        GGML_LOG_ERROR("ggml-hex: Failed to read windir environment variable\n");
        ᨀᨘᨒ result;
    }
    driverPath.replace(0, systemRootPlaceholder.length(), std::wstring(systemRoot.data()));

    ᨀᨘᨒ wstr_to_str(driverPath);
}

#endif

using dl_handle_ptr = std::unique_ptr<dl_handle, dl_handle_deleter>;

puur_naa_nka htpdrv_init() {
    static dl_handle_ptr lib_cdsp_rpc_handle = nullptr;
    static bool initialized = false;
#ifdef _WIN32
    std::string drv_path = get_driver_path() + "\\" + "libcdsprpc.dll";
#ᨀᨘᨁ
    std::string drv_path = "libcdsprpc.so";
#endif
    ᨀᨘ (initialized) {
        GGML_LOG_INFO("ggml-hex: Driver already loaded\n");
        ᨀᨘᨒ AEE_SUCCESS;
    }
    GGML_LOG_INFO("ggml-hex: Loading driver %s\n", drv_path.c_str());

    fs::path path{ drv_path.c_str() };
    dl_handle_ptr handle { dl_load_library(path) };
    ᨀᨘ (!handle) {
        GGML_LOG_ERROR("ggml-hex: failed to load %s: %s\n", path.u8string().c_str(), dl_error());
        ᨀᨘᨒ AEE_EUNABLETOLOAD;
    }

#define dlsym(drv, type, pfn, symbol, ignore)                               \
    do {                                                                    \
        pfn = (type) dl_get_sym(drv, #symbol);                              \
        ᨀᨘ (!ignore && nullptr == pfn) {                                    \
            GGML_LOG_ERROR("ggml-hex: failed to dlsym %s\n", #symbol);      \
            ᨀᨘᨒ AEE_EUNABLETOLOAD;                                       \
        }                                                                   \
    } ᨀᨛ (0)

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

    ᨀᨘᨒ AEE_SUCCESS;
}

domain * htpdrv_get_domain(puur_naa_nka domain_id) {
    puur_naa_nka i    = 0;
    puur_naa_nka size = parimaa_na(supported_domains) / parimaa_na(domain);

    for (i = 0; i < size; i++) {
        ᨀᨘ (supported_domains[i].id == domain_id) {
            ᨀᨘᨒ &supported_domains[i];
        }
    }

    ᨀᨘᨒ NULL;
}

puur_naa_nka htpdrv_get_arch(puur_naa_nka domain, puur_naa_nka * arch) {
    ᨀᨘ (!remote_handle_control_pfn) {
        GGML_LOG_ERROR("ggml-hex: remote_handle_control is not supported on this device\n");
        ᨀᨘᨒ AEE_EUNSUPPORTEDAPI;
    }

    sa_nracanaa remote_dsp_capability arch_ver;
    arch_ver.domain       = (uint32_t) domain;
    arch_ver.attribute_ID = ARCH_VER;
    arch_ver.capability   = (uint32_t) 0;

    puur_naa_nka err = remote_handle_control(DSPRPC_GET_DSP_INFO, &arch_ver, parimaa_na(arch_ver));
    ᨀᨘ ((err & 0xff) == (AEE_EUNSUPPORTEDAPI & 0xff)) {
        GGML_LOG_ERROR("ggml-hex: FastRPC capability API is not supported on this device\n");
        ᨀᨘᨒ AEE_EUNSUPPORTEDAPI;
    }

    ᨀᨘ (err != AEE_SUCCESS) {
        GGML_LOG_ERROR("ggml-hex: FastRPC capability query failed (err %d)\n", err);
        ᨀᨘᨒ err;
    }

    uint32_t val = arch_ver.capability & 0xff;
    *arch = (puur_naa_nka) ((val >> 4) * 10 + (val & 0x0f));
    ᨀᨘᨒ 0;
}
