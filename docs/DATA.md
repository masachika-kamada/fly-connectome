# データの出どころ

このリポジトリは公開データだけを使っています。どちらも再取得できます。

## 1. コネクトーム — MaleCNS v1.0

- 提供: HHMI Janelia Research Campus / Google Research（2026年9月）
- 取得: neuPrint の公開 Cypher エンドポイント。**認証不要**
  `POST https://neuprint.janelia.org/api/custom/custom`
  `{"dataset": "male-cns:v1.0", "cypher": "..."}`
- 取得するもの: 右半球キノコ体の部分グラフ（ALPN / Kenyon_Cell / MBON / DAN）
- スクリプト: `scripts/fetch_connectome.py` → `data/mb_right.npz`, `data/mb_meta.json`
- ライセンス: CC BY 4.0

引いているのは `class` ラベルで選別した部分グラフだけで、全体（166,700ニューロン）は落としていません。

## 2. 匂いの応答 — DoOR 2.0

- 提供: Münch & Galizia, *Scientific Reports* 6:21841 (2016)
- 取得: https://github.com/ropensci/DoOR.data の `data/` から3ファイル
  - `door_response_matrix.csv` — 691化学物質 × 78受容体のコンセンサス応答
  - `odor.csv` — 物質名と化学クラス
  - `door_mappings.csv` — 受容体 → 糸球体の対応
- コミットしていない（上流のもので、やや大きい）ので、初回は取得が要る

```bash
uv run scripts/fetch_odors.py
```

- ライセンス: CC BY-SA 4.0（DoOR.data のパッケージ記述による）。
  ここから作ったデータ（`data/odor_kc.json`、`data/errand.json` の匂いの部分）も同じ CC BY-SA 4.0 で配布する

## 突き合わせ

DoOR の受容体を糸球体名に変換し、MaleCNS 側の ALPN の糸球体名と照合しています。
**47糸球体が両データで一致**。DA1（フェロモン）、DP1m（酢酸）、DM1/DM2/DM4（果実のエステル）を含みます。

変換は `flymb/door.py` にあり、デモ用の `scripts/build_odor_map.py`（→ `data/odor_kc.json`）と
実験4（`experiments/exp4_real_odors.py`）が同じものを使います。デモの数字と実験の数字が同じ匂いから出るようにするためです。

測定済みの受容体チャンネルが半分以上ある物質だけを残すので、691物質のうち使うのは **67物質**です。
欠測を0で埋めれば数は増やせますが、DoOR の欠測は「反応しない」ではなく「測っていない」なので、そうしていません。

## 文献から持ち込んでいるもの

データから導いたのではなく、文献知識として持ち込んでいる分類が2つあります。
結果の分類・表示とデモの文言に使うほか、VP 系＝温度・湿度という分類だけは、実験1の `olfactory` モデルで
「合成の匂いをどのチャンネルに入れるか」を決めるのに使っています。

- 糸球体の役割（`flymb/odors.py`）: VP 系は温度・湿度（Marin et al. 2020）、DP1m は酢酸、DA1 は cVA フェロモン、など
- 区画の価値（`flymb/capacity.py`）: PPL1 のドーパミンが届く区画は罰、PAM が届く区画は報酬（Aso et al., eLife 2014）

## 生成物

`data/mb_right.npz`, `data/mb_meta.json`, `data/odor_kc.json`, `results/*.json`,
`figures/*.svg`, `data/errand.json`, `dist/*.html` は全部生成物です。README の手順で作り直せます。
`data/errand_controls.json`（実験5用の追加の張り替え配線）は大きいのでコミットしていません。`scripts/build_errand.py` が作ります。
