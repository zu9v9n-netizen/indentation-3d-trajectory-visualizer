# Indentation 3D Trajectory Visualizer

CSV の `time / load / displacement` から trial を検出し、contact start を基準に荷重オフセット補正と変位ゼロ補正を行い、3D 軌跡プロット用のデータと PNG を出力します。

速度一致、仕事量、room/cool 比較、trial 除外などの解析は入れていません。3D 可視化に必要な補正済み時系列だけに絞っています。

## インストール

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e .
```

## 使い方

```bash
indentation-3d analyze path\to\measurement.csv --subject sample01 --condition room
```

列名を自動推定できない場合は、列名または 0 始まりの列番号を指定できます。

```bash
indentation-3d analyze measurement.csv --time-column time_s --load-column force_n --displacement-column disp_mm
indentation-3d analyze measurement.csv --load-column 0 --displacement-column 1 --sample-interval-sec 0.0005
```

主な出力先:

```text
outputs/
  3d_visualization/
    batch_YYYYMMDD_HHMMSS/
      corrected_time_series_by_trial.csv
      contact_points.csv
      analysis_settings.json
      plots/
        3d/
        overview/
```

## 出力される主な列

- `relative_time_s`
- `indentation_mm`
- `load_zeroed_N`
- `load_smooth_zeroed_N`
- `contact_start_index`
- `contact_start_time_s`
- `contact_displacement_mm`
- `load_offset_N`
- `event_quality_label`

基本の 3D 軸は `relative_time_s`, `indentation_mm`, `load_zeroed_N` です。ノイズを抑えて確認したい場合は `load_smooth_zeroed_N` も出力されます。

## 処理の流れ

1. CSV を複数エンコーディングで読み込み
2. `time / load / displacement` を抽出
3. time がない場合はサンプリング間隔から生成
4. 荷重を移動平均で軽く平滑化
5. 検出用の荷重ベースラインを推定
6. しきい値を超える区間を trial 候補として検出
7. 各 trial の contact start を探索
8. contact start 前の荷重中央値で荷重オフセット補正
9. contact start の変位を 0 として変位ゼロ補正
10. 補正済み時系列、contact point、3D/overview プロットを保存

## 開発用テスト

```bash
python -m unittest discover -s tests
```
