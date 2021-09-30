#ifndef TRT_INFER_CRC_H
#define TRT_INFER_CRC_H

#include <cstddef> // IWYU pragma: keep
#include <cstdint> // IWYU pragma: keep

namespace sample
{
namespace crc
{

//!
//! \brief Compute the CRC-32 checksum of a memory region.
//!
//! \param buffer Pointer to start of memory region
//! \param size Size of memory region in bytes
//!
//! \returns CRC-32 checksum
//!
uint32_t crc32(const void* buffer, size_t size);

} // namespace crc
} // namespace sample

#endif
