#ifndef DLPACK_DLPACK_H_
#define DLPACK_DLPACK_H_

#ifdef __cplusplus
#define DLPACK_EXTERN_C extern "C"
#else
#define DLPACK_EXTERN_C
#endif

#define DLPACK_MAJOR_VERSION 1

#define DLPACK_MINOR_VERSION 3

#ifdef _WIN32
#ifdef DLPACK_EXPORTS
#define DLPACK_DLL __declspec(dllexport)
#else
#define DLPACK_DLL __declspec(dllimport)
#endif
#else
#define DLPACK_DLL
#endif

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  uint32_t major;
  uint32_t minor;
} DLPackVersion;

#ifdef __cplusplus
typedef enum : int32_t {
#else
typedef enum {
#endif
  kDLCPU = 1,
  kDLCUDA = 2,
  kDLCUDAHost = 3,
  kDLOpenCL = 4,
  kDLVulkan = 7,
  kDLMetal = 8,
  kDLVPI = 9,
  kDLROCM = 10,
  kDLROCMHost = 11,
  kDLExtDev = 12,
  kDLCUDAManaged = 13,
  kDLOneAPI = 14,
  kDLWebGPU = 15,
  kDLHexagon = 16,
  kDLMAIA = 17,
  kDLTrn = 18,
} DLDeviceType;

typedef struct {
  DLDeviceType device_type;
  int32_t device_id;
} DLDevice;

typedef enum {
  kDLInt = 0U,
  kDLUInt = 1U,
  kDLFloat = 2U,
  kDLOpaqueHandle = 3U,
  kDLBfloat = 4U,
  kDLComplex = 5U,
  kDLBool = 6U,
  kDLFloat8_e3m4 = 7U,
  kDLFloat8_e4m3 = 8U,
  kDLFloat8_e4m3b11fnuz = 9U,
  kDLFloat8_e4m3fn = 10U,
  kDLFloat8_e4m3fnuz = 11U,
  kDLFloat8_e5m2 = 12U,
  kDLFloat8_e5m2fnuz = 13U,
  kDLFloat8_e8m0fnu = 14U,
  kDLFloat6_e2m3fn = 15U,
  kDLFloat6_e3m2fn = 16U,
  kDLFloat4_e2m1fn = 17U,
} DLDataTypeCode;

typedef struct {
  uint8_t code;
  uint8_t bits;
  uint16_t lanes;
} DLDataType;

typedef struct {
  void *data;
  DLDevice device;
  int32_t ndim;
  DLDataType dtype;
  int64_t *shape;
  int64_t *strides;
  uint64_t byte_offset;
} DLTensor;

typedef struct DLManagedTensor {
  DLTensor dl_tensor;
  void *manager_ctx;
  void (*deleter)(struct DLManagedTensor *self);
} DLManagedTensor;

#define DLPACK_FLAG_BITMASK_READ_ONLY (1UL << 0UL)

#define DLPACK_FLAG_BITMASK_IS_COPIED (1UL << 1UL)

#define DLPACK_FLAG_BITMASK_IS_SUBBYTE_TYPE_PADDED (1UL << 2UL)

typedef struct DLManagedTensorVersioned {
  DLPackVersion version;
  void *manager_ctx;
  void (*deleter)(struct DLManagedTensorVersioned *self);
  uint64_t flags;
  DLTensor dl_tensor;
} DLManagedTensorVersioned;

typedef int (*DLPackManagedTensorAllocator)(
    DLTensor *prototype, DLManagedTensorVersioned **out, void *error_ctx,
    void (*SetError)(void *error_ctx, const char *kind, const char *message));

typedef int (*DLPackManagedTensorFromPyObjectNoSync)(
    void *py_object, DLManagedTensorVersioned **out);

typedef int (*DLPackDLTensorFromPyObjectNoSync)(void *py_object, DLTensor *out);

typedef int (*DLPackCurrentWorkStream)(DLDeviceType device_type,
                                       int32_t device_id,
                                       void **out_current_stream);

typedef int (*DLPackManagedTensorToPyObjectNoSync)(
    DLManagedTensorVersioned *tensor, void **out_py_object);

typedef struct DLPackExchangeAPIHeader {
  DLPackVersion version;
  struct DLPackExchangeAPIHeader *prev_api;
} DLPackExchangeAPIHeader;

typedef struct DLPackExchangeAPI {
  DLPackExchangeAPIHeader header;
  DLPackManagedTensorAllocator managed_tensor_allocator;
  DLPackManagedTensorFromPyObjectNoSync managed_tensor_from_py_object_no_sync;
  DLPackManagedTensorToPyObjectNoSync managed_tensor_to_py_object_no_sync;
  DLPackDLTensorFromPyObjectNoSync dltensor_from_py_object_no_sync;
  DLPackCurrentWorkStream current_work_stream;
} DLPackExchangeAPI;

#ifdef __cplusplus
}
#endif
#endif
