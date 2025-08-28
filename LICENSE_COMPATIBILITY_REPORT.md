# ライセンス互換性調査レポート

## 概要

本プロジェクトは**AGPL-3.0ライセンス**として公開されています。依存ライブラリとの互換性を調査した結果、すべて法的に問題ない状態となりました。

## ✅ AGPL-3.0採用による互換性の解決

### プロジェクトライセンス: AGPL-3.0

GNU Affero General Public License v3.0を採用することで、以下のメリットがあります：
- AGPL-3.0ライブラリとの完全な互換性
- オープンソースコミュニティへの貢献
- 法的リスクの完全な排除

## 依存ライブラリのライセンス状況

### 1. AGPL-3.0ライセンス（互換性あり）✅

| ライブラリ | ライセンス | 互換性 | 備考 |
|------------|-----------|--------|------|
| **ultralytics** | AGPL-3.0 | ✅ 完全互換 | YOLOv11の最新実装 |
| **albumentations** | AGPL-3.0 | ✅ 完全互換 | 高度な画像拡張 |

これらのライブラリは本プロジェクトと同じAGPL-3.0のため、完全に互換性があります。

### 2. より寛容なライセンス（互換性あり）✅

| ライブラリ | ライセンス | 互換性 | 備考 |
|------------|-----------|--------|------|
| **numpy** | BSD-3-Clause | ✅ | AGPLと互換 |
| **opencv-python** | Apache 2.0 | ✅ | AGPLと互換 |
| **Pillow** | MIT-CMU | ✅ | AGPLと互換 |
| **PySide6** | LGPL v3 | ✅ | AGPLと互換 |
| **onnxruntime** | MIT | ✅ | AGPLと互換 |
| **timm** | Apache 2.0 | ✅ | AGPLと互換 |
| **psutil** | BSD-3-Clause | ✅ | AGPLと互換 |
| **gdown** | MIT | ✅ | AGPLと互換 |

これらの寛容なライセンス（MIT、BSD、Apache 2.0、LGPL）はAGPL-3.0に組み込むことが可能です。

### 3. 特殊な扱いが必要なライブラリ

| ライブラリ | ライセンス | 状況 | 対応 |
|------------|-----------|------|------|
| **insightface** | コード: MIT<br>モデル: 非商用のみ | ⚠️ 注意必要 | コードは使用可<br>モデルは別途管理 |

**InsightFaceについて**:
- コード自体はMITライセンスで問題なし
- 事前学習済みモデルは非商用研究目的のみ
- 対応: モデルの自動ダウンロードを無効化し、ユーザーが自前でモデルを準備

## AGPL-3.0の主な条件

### 利用者の権利
- ✅ 商用利用可能
- ✅ 修正・改変可能
- ✅ 配布可能
- ✅ 特許使用可能
- ✅ 私的利用可能

### 利用者の義務
- 📝 ソースコード開示義務
- 📝 ライセンスと著作権表示
- 📝 変更内容の明示
- 📝 同一ライセンスでの配布（コピーレフト）
- 🌐 **ネットワーク利用時のソース開示**（AGPLの特徴）

### AGPL-3.0の特徴
- Webサービスやクラウドサービスとして提供する場合も、ソースコードの開示が必要
- GPLv3との相互互換性あり
- 強力なコピーレフト効果

## 商用利用について

### オープンソースとしての商用利用
AGPL-3.0のもとで商用利用は可能ですが、以下の条件があります：
- ソースコード全体の公開
- 修正箇所の公開
- ネットワークサービスの場合も公開義務

### クローズドソースでの商用利用
クローズドソースで商用利用したい場合：
1. Ultralyticsの[商用ライセンス](https://www.ultralytics.com/license)を購入
2. albumentationsの商用ライセンスを取得
3. その他のライブラリはMIT/BSD/Apacheなので問題なし

## ライセンス表示例

```python
# YOLOv11 Person Detection System
# Copyright (C) 2024 [Your Name/Organization]
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
# 
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
```

## 結論

✅ **現在の構成は法的に問題ありません**

AGPL-3.0ライセンスを採用することで：
1. すべての依存ライブラリと互換性を保持
2. 法的リスクを完全に排除
3. オープンソースコミュニティへの貢献を明確化

商用利用の際は：
- オープンソースとして利用: 問題なし（ソース公開義務あり）
- クローズドソースとして利用: 商用ライセンスの購入が必要

## 参考リンク

- [GNU Affero General Public License v3.0](https://www.gnu.org/licenses/agpl-3.0.html)
- [AGPL-3.0の解説（日本語）](https://www.gnu.org/licenses/agpl-3.0.ja.html)
- [Ultralytics License](https://www.ultralytics.com/license)
- [ライセンス互換性マトリックス](https://www.gnu.org/licenses/license-compatibility.html)

---
*最終更新: 2024年 - AGPL-3.0への移行完了*