#include "crc32.h"
#include <array>
#include <climits>

// FIXME TRTS-3445: Eliminate the code duplication here and in sample/common/crc32.cpp

namespace sample
{
namespace crc
{

namespace
{

// Number of different possible byte values
constexpr int32_t kBYTE_VALUES = 256;

// Number of bits in a byte
constexpr int32_t kBITS_PER_BYTE = 8;
static_assert(CHAR_BIT == kBITS_PER_BYTE, "Error");

// 32-bit hash value with all bits set
constexpr uint32_t kALL_BITS_SET = 0xFFFFFFFF;

// 32-bit hash value with no bits set
constexpr uint32_t kNO_BITS_SET = 0;

// Official CRC-32 checksum polynomial, in reversed order
constexpr uint32_t kPOLYNOMIAL_REVERSED = 0xEDB88320;

// Precompute table of polynomial bits to flip for incoming bytes
const std::array<uint32_t, kBYTE_VALUES> initTable()
{
    std::array<uint32_t, kBYTE_VALUES> table;
    // Precompute table for bytes
    for (int32_t i = 0; i < kBYTE_VALUES; i++)
    {
        uint32_t value = i;
        for (int32_t j = 0; j < kBITS_PER_BYTE; j++)
        {
            uint32_t mask = (value & 1U) ? kALL_BITS_SET : kNO_BITS_SET;
            // Polynomial is stored in reversed format
            value = (value >> 1U) ^ (kPOLYNOMIAL_REVERSED & mask);
        }
        table[i] = value;
    }
    return table;
}

} // namespace

uint32_t crc32(const void* buffer, size_t size)
{
    // Implementation is original from high-level description in wikipedia
    // https://en.wikipedia.org/wiki/Cyclic_redundancy_check#CRC-32_algorithm

    static const std::array<uint32_t, kBYTE_VALUES> table = initTable();

    const auto* ptr = static_cast<const uint8_t*>(buffer);
    uint32_t sum = kALL_BITS_SET;
    while (size--)
    {
        uint8_t byte = *ptr++;
        uint8_t index = (sum & 0xffU) ^ byte;
        sum = (sum >> 8U) ^ table[index];
    }
    // Invert bits at end
    return ~sum;
}

} // namespace crc
} // namespace sample
