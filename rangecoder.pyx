from libc.stdlib cimport malloc, free
import cython

cdef int CODE_BITS = 32
cdef unsigned int MAX_VALUE = 0xFFFFFFFF
cdef unsigned int MIN_RANGE = (1 << (CODE_BITS - 8))
cdef unsigned int TOP_VALUE = (1 << (CODE_BITS - 1))
cdef unsigned int FIRST_QUARTER = (1 << (CODE_BITS - 2))

cdef int MAX_TOTAL_COUNT = 4096

cdef unsigned char EOB_SEQUENCE_BITS[16]
cdef int EOB_SEQUENCE_LEN = 16

cdef int EOB_HISTORY_LEN = EOB_SEQUENCE_LEN

cdef void init_eob_sequence() nogil:
    cdef unsigned char temp_seq[16]
    cdef int i
    for i from 0 <= i < 16: temp_seq[i] = 0 if (i % 8) == 7 else 1
    for i from 0 <= i < EOB_SEQUENCE_LEN: EOB_SEQUENCE_BITS[i] = temp_seq[i]

init_eob_sequence()


cdef inline int write_byte(unsigned char byte, unsigned char *buffer, size_t *buffer_ptr, size_t buffer_end) nogil:
    if buffer_ptr[0] >= buffer_end: return -1
    buffer[buffer_ptr[0]] = byte
    buffer_ptr[0] += 1
    return 0

cdef inline unsigned char read_byte(const unsigned char *buffer, size_t *buffer_ptr, size_t buffer_end, int *is_eof) nogil:
    if buffer_ptr[0] >= buffer_end:
        is_eof[0] = 1
        return 0
    is_eof[0] = 0
    cdef unsigned char byte = buffer[buffer_ptr[0]]
    buffer_ptr[0] += 1
    return byte

cdef inline int rc_normalize_encode(unsigned int *low, unsigned int *range, unsigned char *buffer, size_t *buffer_ptr, size_t buffer_end, int *pending_bytes) nogil:
    cdef unsigned char byte_to_output
    cdef int ret = 0
    while range[0] <= MIN_RANGE:
        byte_to_output = low[0] >> (CODE_BITS - 8)
        low[0] <<= 8
        range[0] <<= 8
        if byte_to_output != 0xFF:
            ret = write_byte(byte_to_output, buffer, buffer_ptr, buffer_end)
            if ret != 0: return ret
            while pending_bytes[0] > 0:
                ret = write_byte(0xFF, buffer, buffer_ptr, buffer_end)
                if ret != 0: return ret
                pending_bytes[0] -= 1
        else: pending_bytes[0] += 1
    return 0

cdef inline int rc_normalize_decode(unsigned int *low, unsigned int *range, unsigned int *value, const unsigned char *buffer, size_t *buffer_ptr, size_t buffer_end) nogil:
    cdef unsigned char byte_read
    cdef int is_eof = 0
    while range[0] <= MIN_RANGE:
        value[0] = (value[0] << 8) & MAX_VALUE
        low[0] = (low[0] << 8) & MAX_VALUE
        range[0] <<= 8
        byte_read = read_byte(buffer, buffer_ptr, buffer_end, &is_eof)
        value[0] |= byte_read
    return is_eof

cdef inline void rescale_counts(int *count0, int *count1, int *total_count) nogil:
    count0[0] = max(1, count0[0] // 2)
    count1[0] = max(1, count1[0] // 2)
    total_count[0] = count0[0] + count1[0]

cdef inline int encode_symbol(int symbol, unsigned int *low, unsigned int *range, int *count0, int *count1, int *total_count, unsigned char *buffer, size_t *buffer_ptr, size_t buffer_end, int *pending_bytes) nogil:
    cdef int lps, mps
    cdef int count_lps, count_mps
    cdef unsigned int split
    cdef int ret = 0
    if count0[0] > count1[0]:
        mps = 0; lps = 1
        count_mps = count0[0]; count_lps = count1[0]
    else:
        mps = 1; lps = 0
        count_mps = count1[0]; count_lps = count0[0]
    split = <unsigned int>(((<unsigned long long>range[0]) * count_lps) / total_count[0])
    if symbol == lps: range[0] = split
    else:
        low[0] += split
        range[0] -= split
    if symbol == 0: count0[0] += 1
    else: count1[0] += 1
    total_count[0] += 1
    if total_count[0] > MAX_TOTAL_COUNT: rescale_counts(count0, count1, total_count)
    ret = rc_normalize_encode(low, range, buffer, buffer_ptr, buffer_end, pending_bytes)
    return ret

cdef int encode_data_nogil(const unsigned char *input_data, size_t input_byte_size, size_t input_bit_size, unsigned char *output_buffer, size_t output_capacity, size_t *encoded_byte_size) nogil:
    cdef unsigned int low = 0
    cdef unsigned int range = MAX_VALUE
    cdef size_t buffer_ptr = 0
    cdef int pending_bytes = 0
    cdef int count0 = 1
    cdef int count1 = 1
    cdef int total_count = count0 + count1
    cdef size_t i, j
    cdef int symbol
    cdef int ret = 0
    cdef int k

    for k from 0 <= k < EOB_SEQUENCE_LEN:
        symbol = EOB_SEQUENCE_BITS[k]
        ret = encode_symbol(symbol, &low, &range, &count0, &count1, &total_count, output_buffer, &buffer_ptr, output_capacity, &pending_bytes)
        if ret != 0: return -1

    for i from 0 <= i < input_bit_size:
        j = <size_t>(i / 8)
        k = i % 8
        symbol = (input_data[j] >> (7 - k)) & 1
        ret = encode_symbol(symbol, &low, &range, &count0, &count1, &total_count, output_buffer, &buffer_ptr, output_capacity, &pending_bytes)
        if ret != 0: return -1

    rc_normalize_encode(&low, &range, output_buffer, &buffer_ptr, output_capacity, &pending_bytes)

    ret = write_byte(low >> 24, output_buffer, &buffer_ptr, output_capacity)
    if ret != 0: return -1
    while pending_bytes > 0:
         ret = write_byte(0x00, output_buffer, &buffer_ptr, output_capacity)
         if ret != 0: return ret
         pending_bytes -= 1
    ret = write_byte(low >> 16, output_buffer, &buffer_ptr, output_capacity)
    if ret != 0: return ret

    encoded_byte_size[0] = buffer_ptr

    return 0

cdef inline int decode_symbol(unsigned int *low, unsigned int *range, unsigned int *value, int *count0, int *count1, int *total_count, const unsigned char *buffer, size_t *buffer_ptr, size_t buffer_end) nogil:
    cdef int lps, mps
    cdef int count_lps, count_mps
    cdef unsigned int split
    cdef unsigned int current_split
    cdef int decoded_symbol = -1
    cdef int is_eof = 0

    if count0[0] > count1[0]:
        mps = 0; lps = 1
        count_mps = count0[0]; count_lps = count1[0]
    else:
        mps = 1; lps = 0
        count_mps = count1[0]; count_lps = count0[0]

    split = <unsigned int>(((<unsigned long long>range[0]) * count_lps) / total_count[0])

    current_split = value[0] - low[0]

    if current_split < split:
        decoded_symbol = lps
        range[0] = split
    else:
        decoded_symbol = mps
        low[0] += split
        range[0] -= split
        value[0] -= split

    if decoded_symbol == 0: count0[0] += 1
    elif decoded_symbol == 1: count1[0] += 1

    if total_count[0] > MAX_TOTAL_COUNT: rescale_counts(count0, count1, total_count)

    is_eof = rc_normalize_decode(low, range, value, buffer, buffer_ptr, buffer_end)

    return decoded_symbol

cdef int decode_data_block_nogil(const unsigned char *encoded_data, size_t encoded_size, unsigned char *decoded_buffer, size_t decoded_capacity, size_t *decoded_bit_size, size_t *encoded_bytes_read) nogil:
    cdef unsigned int low = 0
    cdef unsigned int range = MAX_VALUE
    cdef unsigned int value = 0
    cdef size_t buffer_ptr = 0
    cdef int count0 = 1
    cdef int count1 = 1
    cdef int total_count = count0 + count1

    cdef size_t i
    cdef int decoded_symbol = -1
    cdef int is_eof = 0
    cdef int k

    cdef unsigned char eob_history[16]

    cdef int eob_history_idx = 0
    cdef int eob_detected = 0
    cdef int history_start_idx
    cdef size_t byte_idx
    cdef int bit_in_byte_idx

    for i from 0 <= i < CODE_BITS // 8:
        is_eof = 0
        value = (value << 8) | read_byte(encoded_data, &buffer_ptr, encoded_size, &is_eof)
        if is_eof: return -1

    for k from 0 <= k < EOB_HISTORY_LEN: eob_history[k] = 0

    i = 0
    while i < decoded_capacity * 8 and not eob_detected:
        decoded_symbol = decode_symbol(&low, &range, &value, &count0, &count1, &total_count,
                                       encoded_data, &buffer_ptr, encoded_size)

        eob_history[eob_history_idx] = <unsigned char>decoded_symbol
        eob_history_idx = (eob_history_idx + 1) % EOB_HISTORY_LEN

        if i >= EOB_HISTORY_LEN - 1:
            eob_detected = 1
            history_start_idx = (eob_history_idx - EOB_HISTORY_LEN + EOB_HISTORY_LEN) % EOB_HISTORY_LEN
            for k from 0 <= k < EOB_SEQUENCE_LEN:
                if eob_history[(history_start_idx + k) % EOB_HISTORY_LEN] != EOB_SEQUENCE_BITS[k]:
                    eob_detected = 0
                    break

        if eob_detected: break

        byte_idx = <size_t>(i / 8)
        bit_in_byte_idx = i % 8

        if byte_idx >= decoded_capacity: return -2

        if bit_in_byte_idx == 0: decoded_buffer[byte_idx] = 0
        decoded_buffer[byte_idx] |= (decoded_symbol << (7 - bit_in_byte_idx))

        i += 1

    decoded_bit_size[0] = i

    encoded_bytes_read[0] = buffer_ptr

    return 0

def encode(bytes input_data):
    cdef size_t input_byte_size = len(input_data)
    cdef size_t input_bit_size = input_byte_size * 8

    cdef size_t output_capacity = <size_t>(input_byte_size + input_byte_size // 2 + 32)
    if output_capacity < 1024: output_capacity = 1024

    cdef unsigned char *output_buffer = <unsigned char *>malloc(output_capacity)
    if output_buffer is NULL:
        print("メモリ確保に失敗しました")
        return None

    cdef size_t encoded_byte_size = 0
    cdef int ret = 0

    try:
        ret = encode_data_nogil(<const unsigned char *>input_data, input_byte_size, input_bit_size, output_buffer, output_capacity, &encoded_byte_size)
        if ret != 0:
            print(f"エンコード中にエラーが発生しました (コード: {ret})")
            return None

        encoded_data = output_buffer[:encoded_byte_size]
        return encoded_data

    finally: free(output_buffer)

def decode(bytes encoded_data):
    cdef size_t encoded_size = len(encoded_data)
    cdef size_t encoded_bytes_read_total = 0

    decoded_blocks = []

    cdef unsigned char *current_encoded_ptr = <unsigned char *>encoded_data
    cdef size_t current_encoded_size = encoded_size

    cdef size_t decoded_capacity
    cdef unsigned char *decoded_buffer = NULL
    cdef size_t decoded_bit_size = 0
    cdef size_t encoded_bytes_read_block = 0
    cdef int ret = 0
    cdef size_t decoded_byte_size

    while encoded_bytes_read_total < encoded_size:
        decoded_capacity = 1024 * 1024

        decoded_buffer = <unsigned char *>malloc(decoded_capacity)
        if decoded_buffer is NULL:
            print("メモリ確保に失敗しました")
            return None

        ret = decode_data_block_nogil(current_encoded_ptr, current_encoded_size, decoded_buffer, decoded_capacity, &decoded_bit_size, &encoded_bytes_read_block)

        if ret != 0:
            print(f"デコード中にエラーが発生しました (コード: {ret})")
            free(decoded_buffer)
            return None

        decoded_byte_size = <size_t>((decoded_bit_size + 7) / 8)

        if decoded_bit_size > 0:
             decoded_block_bytes = decoded_buffer[:decoded_byte_size]
             decoded_blocks.append(decoded_block_bytes)

        free(decoded_buffer)
        decoded_buffer = NULL

        encoded_bytes_read_total += encoded_bytes_read_block
        current_encoded_ptr += encoded_bytes_read_block
        current_encoded_size -= encoded_bytes_read_block

    return decoded_blocks
