# YOLOv11 リアルタイム人物検出システム

PCカメラからリアルタイムで人物を検出するアプリケーション。最新のYOLOv11モデルを使用。

## 特徴

- **最新技術**: YOLOv11（2025年最新モデル）を使用
- **GUI版提供**: PySide6による直感的な操作インターフェース
- **高速処理**: リアルタイム検出（30+ FPS）
- **モジュール設計**: 拡張性の高いコード構造
- **パフォーマンス監視**: FPSと処理時間の表示
- **インタラクティブ**: 動的な閾値調整

## インストール

```bash
# 仮想環境をアクティベート
source .venv/bin/activate

# 依存関係をインストール（既にインストール済み）
pip install -r requirements.txt
```

## 使用方法

### GUI版（推奨）

```bash
# GUIアプリケーションを起動
python gui_main.py
```

**GUI版の特徴：**
- ボタンによる操作（一時停止、スクリーンショット等）
- スライダーによる信頼度調整
- リアルタイム統計表示
- モデル選択ドロップダウン
- カメラ設定の動的変更

### CUI版（従来版）

```bash
# コマンドライン版を起動
python main.py
```

### CUI版オプション指定

```bash
# 高精度モデルを使用
python main.py --model yolo11x.pt

# 信頼度閾値を変更
python main.py --confidence 0.7

# カメラと解像度を指定
python main.py --camera 0 --width 1920 --height 1080

# 中心点表示を有効化
python main.py --show-center

# デバッグモードで実行
python main.py --debug
```

## 操作方法

### GUI版
マウスでボタンをクリックして操作：
- **一時停止/再生ボタン**: 検出の一時停止と再開
- **スクリーンショットボタン**: 現在の画面を保存
- **信頼度スライダー**: リアルタイムで閾値調整
- **モデル選択**: ドロップダウンからモデル変更
- **メニューバー**: ファイル操作、表示設定、ヘルプ

### CUI版（キーボード操作）
| キー | 機能 |
|------|------|
| `q` / `ESC` | 終了 |
| `p` | 一時停止/再開 |
| `s` | スクリーンショット保存 |
| `+` | 信頼度閾値を上げる |
| `-` | 信頼度閾値を下げる |
| `r` | 統計をリセット |
| `c` | 中心点表示のON/OFF |

## プロジェクト構造

```
person-face-age-gender-detector/
├── gui_main.py            # GUI版エントリーポイント
├── main.py                # CUI版エントリーポイント
├── requirements.txt       # 依存関係
├── src/
│   ├── core/             # コア機能
│   │   ├── detector.py   # 人物検出エンジン
│   │   └── camera.py     # カメラ制御
│   ├── gui/              # GUI関連（PySide6）
│   │   ├── windows/      # ウィンドウクラス
│   │   │   └── main_window.py
│   │   ├── widgets/      # UIウィジェット
│   │   │   ├── video_display.py
│   │   │   └── control_panel.py
│   │   ├── workers/      # ワーカースレッド
│   │   │   └── yolo_worker.py
│   │   └── README.md     # GUI構造ドキュメント
│   ├── ui/               # CUI版UI関連
│   │   └── visualizer.py # 検出結果の可視化
│   └── utils/            # ユーティリティ
│       └── performance.py # パフォーマンス監視
├── debug/                # デバッグ・テスト用
│   ├── test_detector.py  # システムテスト
│   └── test_gui.py      # GUIテスト
├── docs/                 # ドキュメント
│   ├── model_evaluation.md # モデル評価レポート
│   └── optimization_guide.md # 最適化ガイド
└── .venv/                # Python仮想環境
```

## モデルバリエーション

| モデル | mAP | 速度 | 用途 |
|--------|-----|------|------|
| yolo11n.pt | 39.5% | 最速 | リアルタイム重視 |
| yolo11s.pt | 47.0% | 高速 | バランス型 |
| yolo11m.pt | 51.5% | 中速 | 精度重視 |
| yolo11l.pt | 53.4% | 低速 | 高精度 |
| yolo11x.pt | 54.7% | 最遅 | 最高精度 |

## 技術仕様

- **物体検出**: YOLOv11（Ultralytics）
- **映像処理**: OpenCV
- **深層学習**: PyTorch（CPU版）
- **Python**: 3.12+

## トラブルシューティング

### カメラが認識されない場合
```bash
# 別のカメラインデックスを試す
python main.py --camera 1
```

### FPSが低い場合
```bash
# 軽量モデルを使用
python main.py --model yolo11n.pt

# 解像度を下げる
python main.py --width 640 --height 480
```

## ライセンス

MIT License