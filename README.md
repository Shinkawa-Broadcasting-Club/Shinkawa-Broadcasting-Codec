# Shinkawa Broadcasting Codecとは?
局内での番組制作時に、オリジナル+プロキシによる大量の領域占領、エンコードの遅さを改善すべく作られたコーデックです。
## 特徴
H.264などの既存のコーデックのほとんどは、ブロック単位での動き補償やフレーム内マッチングに膨大な演算量が使用されているため、演算量削減のためにSBCでは次のような特徴を持っています。

・時間軸方向にも周波数変換を適用、動き補償を不要に

・メディアンカットアルゴリズムによる最適な丸め値の選択、マッチングなしでも高圧縮を達成可能に

・フレーム間のブレンドノイズを防ぐため時間軸方向にはDCT(離散余弦変換)の代わりにDHT(離散アダマール変換)を使用、品質向上を可能に
## 注意
mainは参照用実装です。参照用のため、非常に遅い実装であることに注意してください。
