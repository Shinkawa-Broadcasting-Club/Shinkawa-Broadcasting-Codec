# cython: boundscheck=False, wraparound=False, cdivision=True, nonecheck=False
# setuptools: language = c++

from libc.stdint cimport uint8_t, uint16_t, uint32_t, uint64_t
from libc.stdlib cimport malloc, free, realloc

# レンジコーダーの状態を定義する構造体
cdef struct RangeCoderState:
    uint64_t low      # レンジの下限
    uint64_t range    # レンジの大きさ
    uint64_t buffer   # デコード時の入力バッファ
    uint32_t bits_to_follow # 保留ビットの数 (エンコード用)
    uint32_t buffer_fullness # デコード時のバッファの現在の有効ビット数
    uint8_t* output_buffer # エンコード時の出力バッファ
    uint64_t output_pos # 出力バッファの現在位置
    uint64_t output_buffer_size # 出力バッファのサイズ
    uint8_t* input_buffer # デコード時の入力バッファ
    uint64_t input_pos # 入力バッファの現在位置
    uint64_t input_buffer_size # 入力バッファのサイズ

# 確率モデルの状態を定義する構造体
# シンプルな1ビットコンテキスト（直前のビット）を使用
cdef struct ProbabilityModel:
    uint32_t counts[2][2] # counts[context][symbol]
    uint32_t total_counts[2] # total_counts[context]
    # カウントがこの閾値を超えたらスケーリングを行う
    # uint33_tは存在しないため、uint64_tを使用
    uint64_t MAX_TOTAL_COUNT # カウントのリセット/スケーリング閾値

# デコードされた単一ブロックのデータとサイズを保持する構造体
cdef struct DecodedBlock:
    uint8_t* data # デコードされたデータのポインタ (mallocされる)
    uint64_t size # デコードされたデータのサイズ

# デコードされた全てのブロックの情報を保持する構造体 (decode関数が返す型)
cdef struct DecodedBlocksInfo:
    DecodedBlock* blocks # DecodedBlock構造体の配列へのポインタ (mallocされる)
    uint64_t num_blocks # ブロック数

# 定数定義
cdef uint32_t TOP = 0x01000000 # レンジの上限 (2^24)
cdef uint32_t NORM_THRESHOLD = 0x00800000 # 正規化の閾値 (TOP / 2)
cdef uint32_t SHIFT = 24 # レンジのビット幅
cdef uint32_t EOB_MARKER_BITS = 16 # EOBマーカーのビット数
cdef uint16_t EOB_MARKER = 0x0001 # EOBマーカーのパターン (例: ...0001)

# 確率モデルの初期化
cdef inline void init_model(ProbabilityModel* model) nogil:
    cdef int i, j
    for i in range(2):
        for j in range(2):
            model.counts[i][j] = 1 # 各シンボル、各コンテキストで初期カウントを1とする (エスケープ処理の簡易版)
        model.total_counts[i] = 2 # 初期合計カウント

    # より大きな閾値でスケーリング頻度を減らす
    model.MAX_TOTAL_COUNT = 65536 # 例として、最大合計カウントを設定

# 確率モデルの更新とスケーリング
cdef inline void update_model(ProbabilityModel* model, uint8_t context, uint8_t symbol) nogil:
    model.counts[context][symbol] += 1
    model.total_counts[context] += 1

    # 合計カウントが閾値を超えたらスケーリング
    if model.total_counts[context] > model.MAX_TOTAL_COUNT:
        # 各カウントを半分にし、最低1を保証
        model.counts[context][0] = (model.counts[context][0] >> 1) + 1
        model.counts[context][1] = (model.counts[context][1] >> 1) + 1
        model.total_counts[context] = model.counts[context][0] + model.counts[context][1]


# レンジコーダーの初期化 (エンコード用)
cdef inline int init_encoder(RangeCoderState* state, uint64_t initial_buffer_size) nogil:
    state.low = 0
    state.range = <uint64_t>TOP
    state.bits_to_follow = 0
    state.output_pos = 0
    state.output_buffer_size = initial_buffer_size
    state.output_buffer = <uint8_t*>malloc(initial_buffer_size)
    if state.output_buffer == NULL: return -1 # エラーコードを返す
    return 0 # 成功

# レンジコーダーの初期化 (デコード用)
cdef inline void init_decoder(RangeCoderState* state, uint8_t* input_buffer, uint64_t input_buffer_size) nogil:
    state.low = 0
    state.range = <uint64_t>TOP
    state.buffer = 0
    state.buffer_fullness = 0
    state.input_buffer = input_buffer
    state.input_pos = 0
    state.input_buffer_size = input_buffer_size

    # 最初の数バイトをバッファに読み込む (最低8バイト、または入力の最後まで)
    cdef int bytes_to_read = 8, i # 64ビットバッファを満たすために8バイト読み込む
    if state.input_buffer_size < bytes_to_read: bytes_to_read = state.input_buffer_size

    for i in range(bytes_to_read):
        state.buffer = (state.buffer << 8) | state.input_buffer[state.input_pos]
        state.input_pos += 1
        state.buffer_fullness += 8

    # バッファが完全に満たされていない場合、残りをゼロでパディング (デコードの終端処理のため)
    if state.buffer_fullness < 64:
         state.buffer <<= (64 - state.buffer_fullness)
         state.buffer_fullness = 64


# バイトを出力バッファに書き込む (エンコード用)
cdef inline int write_byte(RangeCoderState* state, uint8_t byte) nogil:
    cdef uint8_t* new_buffer
    # バッファのサイズが足りなければ拡張 (再アロケーション)
    if state.output_pos >= state.output_buffer_size:
        state.output_buffer_size *= 2 # 例としてサイズを倍にする
        # reallocは失敗した場合NULLを返す可能性があるためチェックが必要
        new_buffer = <uint8_t*>realloc(state.output_buffer, state.output_buffer_size)
        if new_buffer == NULL:
            # メモリ割り当て失敗
            state.output_buffer_size = 0 # サイズをリセットしてエラー状態を示す
            return -1 # エラーコードを返す
        state.output_buffer = new_buffer

    state.output_buffer[state.output_pos] = byte
    state.output_pos += 1
    return 0 # 成功

# 保留ビットを処理してバイトを出力 (エンコード用)
cdef inline int follow_bits(RangeCoderState* state, uint8_t bit) nogil:
    if write_byte(state, bit) != 0: return -1 # エラー伝播
    while state.bits_to_follow > 0:
        if write_byte(state, 1 - bit) != 0: return -1 # エラー伝播
        state.bits_to_follow -= 1
    return 0 # 成功

# 正規化処理 (エンコード用)
cdef inline int normalize_encoder(RangeCoderState* state) nogil:
    while state.range <= NORM_THRESHOLD:
        if state.low < <uint64_t>NORM_THRESHOLD:
            if follow_bits(state, 0) != 0: return -1 # エラー伝播
        else:
            state.low -= <uint64_t>NORM_THRESHOLD
            if follow_bits(state, 1) != 0: return -1 # エラー伝播

        state.low <<= SHIFT
        state.range <<= SHIFT
    return 0 # 成功

# ビットをエンコード
cdef inline int encode_bit(RangeCoderState* state, ProbabilityModel* model, uint8_t context, uint8_t symbol) nogil:
    cdef uint64_t range_for_symbol
    cdef uint32_t prob_symbol0 = model.counts[context][0]
    cdef uint32_t total_prob = model.total_counts[context]

    # 確率に基づいてレンジを分割
    # total_probが0になることはinit_modelとupdate_modelの実装上ないはずですが、念のためチェック
    if total_prob == 0: return -1 # エラーコードを返す

    range_for_symbol = state.range / total_prob

    if symbol == 0: state.range = range_for_symbol * prob_symbol0
    else:
        state.low += range_for_symbol * prob_symbol0
        state.range = state.range - range_for_symbol * prob_symbol0

    # 確率モデルを更新
    update_model(model, context, symbol)

    # 正規化
    if normalize_encoder(state) != 0: return -1 # エラー伝播
    return 0 # 成功

# エンコードのフラッシュ処理
cdef inline int flush_encoder(RangeCoderState* state) nogil:
    # 最後のlowの値を処理
    state.bits_to_follow += 1 # 最後のビット
    if state.low < <uint64_t>NORM_THRESHOLD:
        if follow_bits(state, 0) != 0: return -1 # エラー伝播
    else:
        if follow_bits(state, 1) != 0: return -1 # エラー伝播

    # 保留ビットを全て出力
    while state.bits_to_follow > 0:
        if follow_bits(state, 0) != 0: return -1 # エラー伝播
        state.bits_to_follow -= 1
    return 0 # 成功

# バイトを入力バッファから読み込む (デコード用)
# 入力終了時は0を返すのではなく、別途フラグなどで通知する方が安全ですが、
# ここではシンプルにinput_posをチェックすることで入力終了を判定します。
cdef inline uint8_t read_byte(RangeCoderState* state) nogil:
    cdef uint8_t byte
    if state.input_pos < state.input_buffer_size:
        byte = state.input_buffer[state.input_pos]
        state.input_pos += 1
        return byte
    else: return 0

# 正規化処理 (デコード用)
cdef inline void normalize_decoder(RangeCoderState* state) nogil:
    while state.range <= NORM_THRESHOLD:
        state.low <<= SHIFT
        state.range <<= SHIFT
        # バッファから新しいビットを読み込む
        # バッファが空になったら入力から読み込む
        if state.buffer_fullness < SHIFT: # SHIFT (24)ビット未満になったら補充
             state.buffer = (state.buffer << 8) | read_byte(state) # 8ビットずつ読み込む
             state.buffer_fullness += 8

        # バッファからSHIFTビットを取り出す
        # state.buffer = (state.buffer << SHIFT) | (<uint64_t>read_byte(state) << (SHIFT - 8)) # 8ビットずつ読み込む
        # state.buffer_fullness += SHIFT # バッファのビット数を更新


# ビットをデコード
cdef inline uint8_t decode_bit(RangeCoderState* state, ProbabilityModel* model, uint8_t context) nogil:
    cdef uint64_t range_for_symbol
    cdef uint32_t prob_symbol0 = model.counts[context][0]
    cdef uint32_t total_prob = model.total_counts[context]
    cdef uint8_t symbol

    # total_probが0になることはinit_modelとupdate_modelの実装上ないはずですが、念のためチェック
    if total_prob == 0: return 0

    range_for_symbol = state.range / total_prob

    # バッファの値がどのシンボルに対応するかを判定
    # バッファの上位SHIFTビットを使用
    cdef uint64_t buffer_top_bits = state.buffer >> (64 - SHIFT) # バッファの上位SHIFTビット

    if buffer_top_bits < (state.low >> (64 - SHIFT)) + (range_for_symbol >> (64 - SHIFT)) * prob_symbol0:
        symbol = 0
        state.range = range_for_symbol * prob_symbol0
    else:
        symbol = 1
        state.low += range_for_symbol * prob_symbol0
        state.range = state.range - range_for_symbol * prob_symbol0

    # 確率モデルを更新
    update_model(model, context, symbol)

    # 正規化
    normalize_decoder(state)

    return symbol

# バイナリデータをエンコード (nogil対応)
# 戻り値はmallocされたバッファへのポインタ。呼び出し元でfreeする必要がある。
cdef inline uint8_t* encode(uint8_t* input_data, uint64_t input_size, uint64_t* output_size) nogil:
    cdef RangeCoderState state
    cdef ProbabilityModel model
    cdef uint64_t i
    cdef uint8_t current_context = 0 # 初期コンテキスト (例: 0)

    # 初期バッファサイズを適切に設定 (入力サイズの1.5倍など、圧縮率に応じて調整)
    if init_encoder(&state, input_size * 2) != 0:
        output_size[0] = 0
        return NULL # メモリ割り当て失敗

    init_model(&model)
    cdef uint8_t byte
    cdef int bit_pos
    cdef uint8_t bit
    cdef uint16_t eob_pattern
    # 入力データをエンコード
    for i in range(input_size):
        # 各バイトを8ビットに分解してエンコード
        byte = input_data[i]
        for bit_pos in range(7, -1, -1): # MSBから処理
            bit = (byte >> bit_pos) & 1
            if encode_bit(&state, &model, current_context, bit) != 0:
                # エラー発生、バッファを解放してNULLを返す
                free(state.output_buffer)
                output_size[0] = 0
                return NULL
            current_context = bit # コンテキストを更新 (直前のビット)

    # EOBマーカーをエンコード
    eob_pattern = EOB_MARKER
    for bit_pos in range(EOB_MARKER_BITS - 1, -1, -1):
        bit = (eob_pattern >> bit_pos) & 1
        if encode_bit(&state, &model, current_context, bit) != 0:
            # エラー発生
            free(state.output_buffer)
            output_size[0] = 0
            return NULL
        current_context = bit # コンテキストを更新

    # エンコードのフラッシュ
    if flush_encoder(&state) != 0:
        # エラー発生
        free(state.output_buffer)
        output_size[0] = 0
        return NULL

    # 最終的な出力サイズを設定
    output_size[0] = state.output_pos

    return state.output_buffer

# バイナリデータをデコード (nogil対応)
# EOBマーカーで区切られた複数のデータブロックをデコードし、C言語レベルの構造体で返す
# 呼び出し元 (ラッパー関数) で DecodedBlocksInfo 構造体と内部のデータバッファを解放する必要がある
cdef inline DecodedBlocksInfo* decode(uint8_t* input_data, uint64_t input_size) nogil:
    cdef RangeCoderState state
    cdef ProbabilityModel model
    cdef uint8_t current_context = 0 # 初期コンテキスト
    cdef uint8_t* current_block_bits_buffer = NULL # 現在のブロックのビットを格納するCバッファ
    cdef uint64_t current_block_bits_size = 0 # 現在のブロックのビット数
    cdef uint64_t current_block_bits_buffer_capacity = 0 # 現在のブロックのビットバッファ容量

    cdef DecodedBlocksInfo* result = NULL # 結果を格納する構造体
    cdef DecodedBlock* decoded_blocks_array = NULL # デコードされたブロック情報を格納する配列
    cdef uint64_t num_blocks = 0 # デコードされたブロック数
    cdef uint64_t decoded_blocks_array_capacity = 0 # decoded_blocks_array の容量

    # 結果構造体を初期化
    result = <DecodedBlocksInfo*>malloc(sizeof(DecodedBlocksInfo))
    if result == NULL: return NULL # メモリ割り当て失敗

    result.blocks = NULL
    result.num_blocks = 0

    # デコード開始
    init_decoder(&state, input_data, input_size)
    init_model(&model)
    cdef uint8_t decoded_bit
    cdef uint8_t* new_buffer
    cdef uint64_t i
    cdef uint16_t potential_eob
    cdef int bit_pos
    cdef uint64_t block_data_bits_size
    cdef uint64_t block_data_bytes_size
    cdef DecodedBlock* new_array
    cdef uint64_t byte_idx
    cdef uint8_t current_byte
    cdef int bit_count
    cdef uint64_t bit_idx
    # デコードループ
    # 入力データがなくなるまで、またはデコード不能になるまで続ける
    while state.input_pos < state.input_buffer_size or state.buffer_fullness > 0:
        decoded_bit = decode_bit(&state, &model, current_context)
        current_context = decoded_bit # コンテキストを更新

        # デコードされたビットを現在のブロックバッファに追加
        if current_block_bits_size >= current_block_bits_buffer_capacity:
            # バッファ容量が足りなければ拡張
            current_block_bits_buffer_capacity = current_block_bits_buffer_capacity * 2 if current_block_bits_buffer_capacity > 0 else 1024 # 初期容量
            new_buffer = <uint8_t*>realloc(current_block_bits_buffer, current_block_bits_buffer_capacity)
            if new_buffer == NULL:
                # メモリ割り当て失敗、エラー処理
                free(current_block_bits_buffer)
                # これまでにデコードしたブロックのデータも解放する必要がある
                for i in range(num_blocks):
                    free(decoded_blocks_array[i].data)
                free(decoded_blocks_array)
                free(result)
                return NULL
            current_block_bits_buffer = new_buffer

        current_block_bits_buffer[current_block_bits_size] = decoded_bit
        current_block_bits_size += 1

        # EOBマーカーのチェック
        # EOBマーカーは固定長なので、デコードされたビット列の末尾と比較
        if current_block_bits_size >= EOB_MARKER_BITS:
            potential_eob = 0
            # バッファの末尾からEOB_MARKER_BITS分のビットを取得してuint16_tに変換
            for bit_pos in range(EOB_MARKER_BITS):
                potential_eob = (potential_eob << 1) | current_block_bits_buffer[current_block_bits_size - EOB_MARKER_BITS + bit_pos]

            if potential_eob == EOB_MARKER:
                # EOBマーカーを除去したサイズ
                block_data_bits_size = current_block_bits_size - EOB_MARKER_BITS
                block_data_bytes_size = (block_data_bits_size + 7) // 8 # バイト数 (切り上げ)

                # デコードされたブロックデータを格納
                if num_blocks >= decoded_blocks_array_capacity:
                    # ブロック情報配列の容量が足りなければ拡張
                    decoded_blocks_array_capacity = decoded_blocks_array_capacity * 2 if decoded_blocks_array_capacity > 0 else 8 # 初期容量
                    new_array = <DecodedBlock*>realloc(decoded_blocks_array, decoded_blocks_array_capacity * sizeof(DecodedBlock))
                    if new_array == NULL:
                        # メモリ割り当て失敗、エラー処理
                        free(current_block_bits_buffer)
                        for i in range(num_blocks): free(decoded_blocks_array[i].data)
                        free(decoded_blocks_array)
                        free(result)
                        return NULL
                    decoded_blocks_array = new_array

                # ブロックデータ用のメモリを確保し、ビットをバイトに変換してコピー
                decoded_blocks_array[num_blocks].data = <uint8_t*>malloc(block_data_bytes_size)
                if decoded_blocks_array[num_blocks].data == NULL:
                     # メモリ割り当て失敗、エラー処理
                    free(current_block_bits_buffer)
                    for i in range(num_blocks): free(decoded_blocks_array[i].data)
                    free(decoded_blocks_array)
                    free(result)
                    return NULL

                decoded_blocks_array[num_blocks].size = block_data_bytes_size

                # ビット列をバイト列に変換してコピー
                byte_idx = 0
                current_byte = 0
                bit_count = 0
                for bit_idx in range(block_data_bits_size):
                    current_byte = (current_byte << 1) | current_block_bits_buffer[bit_idx]
                    bit_count += 1
                    if bit_count == 8:
                        decoded_blocks_array[num_blocks].data[byte_idx] = current_byte
                        byte_idx += 1
                        current_byte = 0
                        bit_count = 0

                # 最後のバイトが8ビット未満の場合の処理 (パディングなど)
                if bit_count > 0:
                    current_byte <<= (8 - bit_count) # 残りのビットを左詰め
                    decoded_blocks_array[num_blocks].data[byte_idx] = current_byte


                num_blocks += 1

                # 次のブロックのデコードのために状態をリセット
                init_decoder(&state, input_data + state.input_pos, input_size - state.input_pos) # 残りの入力データで初期化
                init_model(&model) # モデルもリセット
                current_context = 0 # コンテキストもリセット
                # ビットバッファをリセット
                free(current_block_bits_buffer)
                current_block_bits_buffer = NULL
                current_block_bits_size = 0
                current_block_bits_buffer_capacity = 0

                # デコードループを継続して次のブロックを探す
                continue # 次のイテレーションへ

        # EOBが見つからずにデコードが入力データの終端に達した場合
        if state.input_pos >= state.input_buffer_size and state.buffer_fullness == 0:
            # EOBが見つからずにデータが終了した場合、現在のブロックを確定
            # 最後のブロックのデータを処理
            if current_block_bits_size > 0:
                 block_data_bits_size = current_block_bits_size
                 block_data_bytes_size = (block_data_bits_size + 7) // 8 # バイト数 (切り上げ)

                 if num_blocks >= decoded_blocks_array_capacity:
                    decoded_blocks_array_capacity = decoded_blocks_array_capacity * 2 if decoded_blocks_array_capacity > 0 else 8
                    new_array = <DecodedBlock*>realloc(decoded_blocks_array, decoded_blocks_array_capacity * sizeof(DecodedBlock))
                    if new_array == NULL:
                        free(current_block_bits_buffer)
                        for i in range(num_blocks): free(decoded_blocks_array[i].data)
                        free(decoded_blocks_array)
                        free(result)
                        return NULL
                    decoded_blocks_array = new_array

                 decoded_blocks_array[num_blocks].data = <uint8_t*>malloc(block_data_bytes_size)
                 if decoded_blocks_array[num_blocks].data == NULL:
                    free(current_block_bits_buffer)
                    for i in range(num_blocks): free(decoded_blocks_array[i].data)
                    free(decoded_blocks_array)
                    free(result)
                    return NULL
                 decoded_blocks_array[num_blocks].size = block_data_bytes_size

                 byte_idx = 0
                 current_byte = 0
                 bit_count = 0
                 for bit_idx in range(block_data_bits_size):
                    current_byte = (current_byte << 1) | current_block_bits_buffer[bit_idx]
                    bit_count += 1
                    if bit_count == 8:
                        decoded_blocks_array[num_blocks].data[byte_idx] = current_byte
                        byte_idx += 1
                        current_byte = 0
                        bit_count = 0

                 if bit_count > 0:
                    current_byte <<= (8 - bit_count)
                    decoded_blocks_array[num_blocks].data[byte_idx] = current_byte

                 num_blocks += 1

            # 最後のブロックのビットバッファを解放
            free(current_block_bits_buffer)
            current_block_bits_buffer = NULL


            break # デコード終了

    # 結果構造体にデコードされたブロック情報を設定
    result.blocks = decoded_blocks_array
    result.num_blocks = num_blocks

    return result

# バイナリデータをデコード (Pythonラッパー)
# decode 関数によって malloc されたメモリを解放する責任を持つ
def decode_data(encoded_data: bytes) -> list:
    cdef uint8_t* input_buffer = <uint8_t*>encoded_data
    cdef uint64_t input_size = len(encoded_data)
    cdef uint64_t i
    cdef bytes block_bytes
    # nogil 関数を呼び出し
    # decode 関数は GIL を解放して実行される
    cdef DecodedBlocksInfo* decoded_info
    with nogil: # decode の呼び出し中は GIL を解放
        decoded_info = decode(input_buffer, input_size)

    # 結果の処理とメモリ解放
    cdef list decoded_blocks_list = []
    if decoded_info != NULL:
        try:
            for i in range(decoded_info.num_blocks):
                # C バッファから Python bytes オブジェクトを作成
                # decoded_info.blocks[i].data が NULL でないかチェック
                if decoded_info.blocks[i].data != NULL:
                    block_bytes = decoded_info.blocks[i].data[:decoded_info.blocks[i].size]
                    decoded_blocks_list.append(block_bytes)
        finally:
            # C で確保したメモリを解放
            if decoded_info.blocks != NULL: # blocks 配列自体が NULL でないかチェック
                for i in range(decoded_info.num_blocks):
                    free(decoded_info.blocks[i].data) # 各ブロックのデータ
                free(decoded_info.blocks) # DecodedBlock 構造体の配列
            free(decoded_info) # DecodedBlocksInfo 構造体

    return decoded_blocks_list

# バイナリデータをエンコード (Pythonラッパー)
# この関数はPythonオブジェクトを受け取り、Pythonオブジェクトを返すため、nogilではありません。
def encode_data(input_data: bytes) -> bytes:
    cdef uint8_t* input_buffer = <uint8_t*>input_data
    cdef uint64_t input_size = len(input_data)
    cdef uint64_t output_size = 0
    cdef uint8_t* output_buffer = NULL # 初期値をNULLにする

    # nogil関数を呼び出し
    # encode 関数は内部で GIL を解放して実行される
    output_buffer = encode(input_buffer, input_size, &output_size)

    if output_buffer == NULL: raise MemoryError("Failed to allocate memory for encoded data.")

    # CythonのメモリをPythonのbytesオブジェクトにコピー
    # Pythonオブジェクトを操作するため GIL が必要だが、encode 呼び出し後は GIL が再取得されている
    cdef bytes encoded_data = output_buffer[:output_size]

    # Cythonで確保したメモリを解放
    free(output_buffer)

    return encoded_data

