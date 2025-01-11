import wave
import numpy as np
from scipy.signal import correlate

def to_mono_array(wav_data, n_channels):
    """
    ステレオの場合は左右を平均してモノラル化し、モノラル配列を返す。
    """
    if n_channels == 1:
        return wav_data  # 既にモノラル
    else:
        # shape=(総フレーム数, 2) -> 平均 -> shape=(総フレーム数,)
        return wav_data.mean(axis=1)

def find_loop_point_by_xcorr(
    full_data: np.ndarray,   # 音源全体(モノラル, shape=(N,))
    window_data: np.ndarray, # ウィンドウ(モノラル, shape=(W,))
    window_start_sample: int,
    exclude_width_samples: int = 10000
) -> int:
    """
    相互相関を計算し、window_dataと最も似ている位置(index)を返す。
    ただし「window開始位置付近 (window_start_sample)」はマッチ候補から除外する。
    
    - exclude_width_samples: window_start_sample の前後何サンプルを除外するか
                             (デフォルト 10000サンプル=約0.23秒 @44.1kHz)

    戻り値:
        best_match: フルデータ中で window_data と最も似ていると推定される開始サンプル
                    ただし除外範囲内の相関ピークは無効化している
    """
    # 相互相関 (モード 'full' & method='fft' で高速に計算)
    corr = correlate(full_data, window_data, mode='full', method='fft')
    # corrの長さは N + W - 1
    # corr[i] に対応する「full_data 側のマッチ位置」は
    #   match_pos = i - (len(window_data) - 1)
    
    # ここで「window_start_sample を含む近傍領域」だけ相関値を -∞ にする
    # すなわち「自分自身とのマッチ」を除外したい
    # -----------------------------------------------------------
    # window_start_sample に対応する i は
    #   i = window_start_sample + (len(window_data) - 1)
    # となるので、その前後 exclude_width_samples をまとめて除外します
    #
    # 例) i_exclude_min ～ i_exclude_max を -∞ にする
    
    w_len = len(window_data)
    center_i = window_start_sample + (w_len - 1)
    
    i_exclude_min = max(0, center_i - exclude_width_samples)
    i_exclude_max = min(center_i + exclude_width_samples, len(corr) - 1)
    
    corr[i_exclude_min : i_exclude_max + 1] = -np.inf
    # -----------------------------------------------------------
    
    # 除外後のピークを探す
    peak_index = np.argmax(corr)
    
    # peak_index に対応する full_data 側のマッチ開始位置
    best_match = peak_index - (w_len - 1)
    return best_match

def cut_loop_wav_fft_exclude(
    input_wav, 
    output_wav,
    window_start_sec=30.0, 
    window_length_sec=10.0,
    exclude_width_sec=1.0
):
    """
    FFTベースの相互相関を用いてループポイントを探し、
    ただし「window_start_sec 付近のマッチ」は除外してループポイントを決定する。
    
    exclude_width_sec: window_start_sec付近何秒分を無効化するか
                       デフォルト1秒前後を除外
    """
    with wave.open(input_wav, 'rb') as wf:
        n_channels = wf.getnchannels()
        sampwidth  = wf.getsampwidth()
        framerate  = wf.getframerate()
        n_frames   = wf.getnframes()
        
        raw_data = wf.readframes(n_frames)
        
        if sampwidth == 2:
            audio_data = np.frombuffer(raw_data, dtype=np.int16)
        else:
            raise ValueError(f"{sampwidth*8}bitは未対応サンプルです。必要に応じて実装してください。")
        
        audio_data = audio_data.reshape(-1, n_channels)
        # モノラル化
        mono_data = to_mono_array(audio_data, n_channels).astype(np.float32)
        
        # ウィンドウ開始サンプル, 長さ
        window_start_sample  = int(window_start_sec  * framerate)
        window_length_sample = int(window_length_sec * framerate)
        
        if window_start_sample + window_length_sample > len(mono_data):
            raise ValueError("ウィンドウ範囲がファイルサイズを超えています。")
        
        # ウィンドウ抽出
        window_data = mono_data[window_start_sample : window_start_sample + window_length_sample]
        
        # 除外幅(サンプル)
        exclude_width_samples = int(exclude_width_sec * framerate)
        
        best_index = find_loop_point_by_xcorr(
            full_data        = mono_data,
            window_data      = window_data,
            window_start_sample = window_start_sample,
            exclude_width_samples= exclude_width_samples
        )
        
        print(f"相互相関ピーク位置: {best_index} [サンプル] (除外幅=±{exclude_width_samples}サンプル)")
        
        # best_index が負なら先頭より前、末尾を超えればファイル外なので補正
        if best_index < 0:
            print("best_index < 0 なので 0 に調整します。")
            best_index = 0
        elif best_index > len(mono_data):
            print("ファイル末尾を超えています。末尾に調整します。")
            best_index = len(mono_data)
        
        # ループ区間を cut (例: window開始位置～best_index)
        loop_data = mono_data[window_start_sample : best_index]
    
    # 出力
    with wave.open(output_wav, 'wb') as out_wf:
        out_wf.setnchannels(1)            # モノラル出力
        out_wf.setsampwidth(2)           # 16bit
        out_wf.setframerate(framerate)
        
        out_data_int16 = loop_data.astype(np.int16).tobytes()
        out_wf.writeframes(out_data_int16)
    
    print(f"出力完了: {output_wav}")

if __name__ == "__main__":
    input_file  = "input.wav"
    output_file = "output_looped.wav"
    
    cut_loop_wav_fft_exclude(
        input_wav         = input_file,
        output_wav        = output_file,
        window_start_sec  = 10.0,
        window_length_sec = 10.0,
        exclude_width_sec = 80.0  # window開始位置の前後1秒は除外
    )
