# Debug Directory

このディレクトリには、デバッグとテスト用のスクリプトが含まれています。

## ファイル

### test_detector.py
システムコンポーネントの動作確認用スクリプト

**実行方法:**
```bash
# プロジェクトルートから
python debug/test_detector.py

# debugディレクトリから
cd debug
python test_detector.py
```

**テスト内容:**
1. PersonDetectorの初期化
2. ダミー画像での検出テスト
3. Visualizerの動作確認
4. PerformanceMonitorの機能テスト

## 注意事項
- これらのスクリプトは開発・デバッグ専用です
- 本番環境では使用しないでください
- テスト実行時はプロジェクトルートの仮想環境を有効化してください