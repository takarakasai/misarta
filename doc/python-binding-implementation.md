# misarta Python Binding — 実装レポート (MVP)

**作成日**: 2026-05-03
**対象**: Phase 1 (MVP) の実装結果
**関連**: [`python-binding-plan.md`](python-binding-plan.md) (設計計画)

---

## 1. サマリ

misarta (Rust 製剛体力学ライブラリ) の Python binding **MVP を完成**。
新規クレート `articara/misarta-py/` を workspace に追加し、PyO3 + maturin で
ネイティブ拡張モジュール `misarta` を提供する。

| 指標 | 結果 |
|------|------|
| ビルド | `maturin develop` 成功 (Rust 1.94 / CPython 3.12 / abi3-py39) |
| Rust LOC | ~480 行 (`misarta-py/src/`) |
| Python LOC | ~140 行 (テストスイート) |
| テスト件数 | **15 件すべて pass** (smoke 5 / FK 6 / dynamics 4) |
| 警告 | 0 (Rust) |

---

## 2. 成果物

### 2.1 新規クレート構成

```
articara/misarta-py/
├── Cargo.toml             # cdylib + PyO3 0.22 + numpy 0.22
├── pyproject.toml         # maturin 設定 (abi3-py39)
├── README.md              # ビルド手順 + 使用例
├── .gitignore
├── src/
│   ├── lib.rs             # #[pymodule] 登録 + 定数
│   ├── conv.rs            # numpy ↔ nalgebra 変換ヘルパ
│   ├── se3.rs             # PySE3
│   ├── model.rs           # PyJointType / PyJointModel / PyModel
│   ├── data.rs            # PyData
│   ├── algorithms.rs      # FK / Jacobian / RNEA / CRBA / ABA
│   └── loaders.rs         # URDF ローダ
├── python/misarta/
│   └── __init__.py        # ネイティブ拡張の再エクスポート
└── tests/
    ├── conftest.py        # 共有 fixture (URDF パス / model)
    ├── test_smoke.py      # モジュール属性 / SE3 / JointType
    ├── test_fk.py         # FK / Jacobian
    └── test_dynamics.py   # CRBA / RNEA / ABA 整合性
```

### 2.2 Workspace への組み込み

`articara/Cargo.toml` の `[workspace] members` に `"misarta-py"` を追加。
**`default-members` には含めない** ことで、通常の `cargo build` で
PyO3 の Python リンクが要求されないようにした。

```toml
[workspace]
members = [".", "plugin-api", "jump-sim-wasm", "jump-sim-runner",
           "misarta", "misarta-py", "quadruped-gait", "xtask"]
default-members = [".", "plugin-api", "jump-sim-wasm", "jump-sim-runner",
                   "misarta", "quadruped-gait"]  # misarta-py は除外
```

`cargo check -p misarta-py` または `maturin develop` で個別にビルドする。

---

## 3. 公開 API (実装済み)

### 3.1 クラス

| Python クラス | 対応する Rust 型 | 主なメソッド/属性 |
|---------------|-----------------|-------------------|
| `misarta.SE3` | `Isometry3<f64>` | `identity()`, `from_homogeneous(M)`, `.rotation`, `.translation`, `.homogeneous()`, `.inverse()`, `__mul__` |
| `misarta.JointType` | `JointType<f64>` enum | `revolute(axis)`, `prismatic(axis)`, `fixed()`, `free_flyer()`, `.kind`, `.nq`, `.nv` |
| `misarta.JointModel` | `JointModel<f64>` | `.name`, `.parent`, `.joint_type`, `.placement` |
| `misarta.Model` | `Arc<Model<f64>>` | `.name`, `.nq`, `.nv`, `.njoints`, `.gravity`, `joint(idx)`, `joint_id(name)`, `joint_names()`, `link_id(name)`, `link_names()` |
| `misarta.Data` | `Data<f64>` | `oMi(i)`, `joint_placement(i)`, `.J`, `body_velocity(i)`, `body_acceleration(i)` |

### 3.2 関数

| 関数 | 対応 Rust 関数 |
|------|----------------|
| `forward_kinematics(model, q) -> Data` | `fk::forward_kinematics` |
| `compute_joint_jacobian(model, q, joint_id, ref_frame=WORLD)` | `jacobian::compute_joint_jacobian{,_local}` |
| `rnea(model, q, v, a)` | `rnea::rnea` |
| `crba(model, q)` | `crba::crba` |
| `aba(model, q, v, tau)` | `aba::aba` |
| `build_model_from_urdf(urdf_str, root=None)` | `urdf::load_urdf_string` |
| `load_urdf(path)` | `urdf::load_urdf` |

### 3.3 定数

| 定数 | 値 |
|------|----|
| `LOCAL` | 0 |
| `WORLD` | 1 |
| `LOCAL_WORLD_ALIGNED` | 2 |

---

## 4. 設計上の決定事項

### 4.1 計画からの変更点 — FK API の純関数化

| 当初案 (Pinocchio 流) | 採用 (misarta 流) |
|-----------------------|-------------------|
| `forward_kinematics(model, data, q)` で `data` を破壊更新 | `forward_kinematics(model, q) -> Data` で fresh Data を返す |

**理由**: misarta 本体の Rust API が純関数 (`fn forward_kinematics(&Model, &[T]) -> Data`)
として書かれているため、Python 側もこれに合わせた方が
- ラッパが薄く保てる
- `&mut PyData` の借用検査と PyO3 の所有権モデルが衝突しない
- ユーザーから見ても misarta の参照透明性原則と一貫する

`misarta.Data(model)` で空 Data を生成することは可能だが、通常は FK が返す
ものをそのまま受け取るのが推奨。

### 4.2 `Arc<Model<f64>>` 共有

`PyModel` は内部で `Arc<Model<f64>>` を保持し、`Clone` も Arc のクローン
(浅いコピー) になる。これにより:
- 大きなモデル (URDF からロードしたもの) を Python から複数の関数に渡しても
  モデル本体のコピーは発生しない
- `Data` がモデルへの安定した参照を保てるが、Python の所有権モデルとも整合
  (Python 側で `del model` しても Data 内の Arc が生き残れば問題ない)

### 4.3 numpy ↔ nalgebra 変換戦略

すべて **コピー渡し** を採用。
- `nalgebra` は column-major、numpy はデフォルト row-major のため、
  ゼロコピー共有は単純化のため見送り
- 入力は `PyReadonlyArray*` から `Vec<f64>` を経由して misarta API のスライス引数に渡す
- 出力は `nalgebra::DMatrix/DVector` を要素列挙して `PyArray2/PyArray1` に詰め直す

将来の最適化余地として `unsafe { from_shape_vec_unchecked(...) }` で
column-major 直接読みは可能だが、Phase 1 ではシンプルさを優先。

### 4.4 abi3-py39

PyO3 features に `abi3-py39` を指定。これにより 1 つの wheel が
**CPython 3.9+ すべてで動作**する (Stable ABI)。CI/CD での配布工数が削減。

### 4.5 PyO3 0.22 と Rust 2024 edition の摩擦

Rust 2024 edition は `unsafe_op_in_unsafe_fn` を warn-by-default にしたため、
PyO3 0.22 のマクロ展開が大量の警告を出す。`misarta-py/src/lib.rs` 冒頭で
`#![allow(unsafe_op_in_unsafe_fn)]` を入れて抑制。
PyO3 0.23+ への移行時にこの allow は不要になる見込み。

---

## 5. 検証

### 5.1 テスト一覧 (15 件すべて pass)

```text
tests/test_smoke.py::test_version                          PASSED
tests/test_smoke.py::test_constants                        PASSED
tests/test_smoke.py::test_se3_identity                     PASSED
tests/test_smoke.py::test_se3_compose_and_inverse          PASSED
tests/test_smoke.py::test_joint_type_factory               PASSED
tests/test_fk.py::test_model_loaded                        PASSED
tests/test_fk.py::test_fk_identity_at_zero                 PASSED
tests/test_fk.py::test_fk_first_joint_homogeneous_is_finite PASSED
tests/test_fk.py::test_fk_changes_with_q                   PASSED
tests/test_fk.py::test_jacobian_shape                      PASSED
tests/test_fk.py::test_jacobian_q_length_validation        PASSED
tests/test_dynamics.py::test_crba_shape_and_symmetry       PASSED
tests/test_dynamics.py::test_rnea_zero_velocity_zero_accel_is_gravity PASSED
tests/test_dynamics.py::test_rnea_aba_roundtrip            PASSED
tests/test_dynamics.py::test_rnea_linearity_in_acceleration PASSED
```

### 5.2 重要な数値整合性チェック

| 検証項目 | 内容 | 許容誤差 |
|----------|------|----------|
| **CRBA 対称正定値性** | `M = M.T`, 全固有値 > 0 | atol=1e-12 |
| **RNEA ↔ ABA 往復** | `aba(q, v, rnea(q, v, a)) ≈ a` | atol=1e-9 |
| **動力学線形性** | `rnea(q, v, a) ≈ M(q)·a + h(q, v)` ここで `h = rnea(q, v, 0)` | atol=1e-9 |
| **SE3 群則** | `T · T⁻¹ ≈ I` | atol=1e-12 |
| **形状検証** | 不正な長さの `q/v/tau` で `ValueError` | — |

これらは misarta 本体の正しさの追加検証にもなっている。

### 5.3 利用 URDF

`misarta/tests/model/urdf/test_robot.urdf` (2 revolute + 1 fixed joint)
を fixture として使用。Rust 単体テストと同じ素材を Python 側からも
参照することで、両言語間で同じデータ素材で動作することを担保。

---

## 6. ビルド/開発ワークフロー

```bash
cd articara/misarta-py
python3 -m venv .venv
source .venv/bin/activate
pip install maturin numpy pytest

maturin develop          # 開発ビルド (debug, ホットリロード)
# または
maturin develop --release  # リリースビルド

pytest tests/             # 全テスト実行
```

`cargo check -p misarta-py` でも Rust 側の文法エラーチェックは可能
(リンクは行われない)。

---

## 7. 既知の制約 / Phase 2 以降への持ち越し

1. **`compute_joint_jacobian` の `LOCAL_WORLD_ALIGNED`**: 現状は WORLD と
   同じ動作。Pinocchio の "linear part を world に直す" セマンティクスは
   misarta 側に対応 API がないため未実装。
2. **型スタブ (`.pyi`)**: 未提供。エディタ補完が弱い。
3. **Pinocchio 数値比較テスト**: pinocchio が入った環境での parity 検証は
   未実装 (Phase 2)。
4. **`forward_kinematics_velocity` / `forward_kinematics_acceleration`**:
   未露出。
5. **`frames`, `centroidal`, `manifold`, `limits`, `ik`, `optimization`,
   `collision`** などは未露出 (計画書の Phase 2-4)。
6. **自動微分**: 未対応。Phase 4 で `rnea_derivatives` / `aba_derivatives`
   を解析微分 API として露出する方針。

---

## 8. 既存ドキュメントへの反映

本実装に伴い以下のドキュメントを更新:

- [`python-binding-plan.md`](python-binding-plan.md) — 実行計画チェックボックスを
  完了状態に更新。FK API の設計変更を反映。
- [`misarta/README.md`](../README.md) — 「Python Bindings」セクションを追加
  し、`misarta-py/` への導線を提供。
- [`articara/README.md`](../../README.md) — workspace 一覧表に `misarta-py`
  を追記。
