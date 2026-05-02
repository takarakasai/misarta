# misarta Python Binding 計画書

**目的**: misarta (Rust 製剛体力学ライブラリ) を Python から利用可能にし、Pinocchio 互換の API を Rust の安全性・性能で提供する。

- **対象クレート**: `articara/misarta-py/` (新規)
- **対象パッケージ**: `misarta` (PyPI)
- **作成日**: 2026-05-03

---

## 1. 設計方針

### 1.1 ツールチェーン

| 採用 | 理由 |
|------|------|
| **PyO3 0.22 + maturin** ✅ | Rust → Python の事実上の標準。`numpy` クレートで `nalgebra ↔ numpy.ndarray` 変換が低コスト |
| **abi3-py39** ✅ | CPython 3.9+ で 1 wheel が動く (Stable ABI)。CI 配布が単純化 |
| cffi / ctypes | Generic を含む API には不向き。却下 |
| cxx + pybind11 | C++ 中継で純 Rust の利点が消える。却下 |

### 1.2 クレート分離

misarta 本体には PyO3 依存を**入れない**。薄い `misarta-py` クレートを workspace に追加することで:

- 本体の `T: RealField` ジェネリクス・自動微分 (Dual) 対応を保てる
- Python wheel ビルドが misarta の通常テストに影響しない
- 将来 `misarta-wasm` 等の他バインディングも同パターンで追加可能

### 1.3 型戦略 — f64 単化

Python 側は **`f64` 単化のみ**を公開。理由:

- Python の Dual 型と Rust の `num_dual::Dual` は ABI が異なり、自動微分の透過は非現実的
- 微分が必要な場合は misarta 内の解析微分 API (`rnea_derivatives`, `aba_derivatives`) を露出することで対応
- PyO3 の型 stub が generics で爆発するのを回避

---

## 2. クレート構成

```
articara/
├── misarta/              # 既存 (純 Rust)
└── misarta-py/           # 新規
    ├── Cargo.toml        # cdylib + pyo3 + numpy
    ├── pyproject.toml    # maturin 設定
    ├── README.md
    ├── src/
    │   ├── lib.rs        # #[pymodule] 宣言
    │   ├── conv.rs       # nalgebra ↔ numpy 変換ヘルパ
    │   ├── se3.rs        # PySE3
    │   ├── model.rs      # PyModel / PyJointModel / PyJointType
    │   ├── data.rs       # PyData
    │   ├── algorithms.rs # fk / jacobian / rnea / crba / aba
    │   └── loaders.rs    # build_model_from_urdf / sdf
    ├── python/misarta/
    │   ├── __init__.py
    │   └── _misarta.pyi  # 型スタブ
    └── tests/
        ├── test_smoke.py
        ├── test_fk.py
        └── test_dynamics.py
```

`articara/Cargo.toml` の `[workspace] members` に `"misarta-py"` を追加する。

---

## 3. 公開 API (フェーズ分け)

### Phase 1 — MVP (今回着手)

Pinocchio との互換性 baseline を取れる最小集合:

```python
import misarta as msrt
import numpy as np

# モデル読込
model = msrt.load_urdf("path/to/robot.urdf")
# あるいは XML 文字列から:
# model = msrt.build_model_from_urdf(urdf_str)

print(model.nq, model.nv, model.name, model.joint_names())

# 順運動学 — misarta は純関数なので fresh Data を返す
q = np.zeros(model.nq)
data = msrt.forward_kinematics(model, q)
T_world = data.oMi(joint_id)             # PySE3 (= nalgebra Isometry3<f64>)

# ヤコビアン (世界座標系)
J = msrt.compute_joint_jacobian(model, q, joint_id, ref_frame=msrt.WORLD)

# 動力学
tau = msrt.rnea(model, q, v, a)          # 逆動力学
M   = msrt.crba(model, q)                # 質量行列
ddq = msrt.aba(model, q, v, tau)         # 順動力学

# SE3 操作
T = msrt.SE3(rotation=R, translation=t)
T2 = T * T                                # 合成
T_inv = T.inverse()
H = T.homogeneous()                       # 4x4 行列
```

> **設計変更ノート**: 当初案では Pinocchio 流の in-place mutation
> (`forward_kinematics(model, data, q)`) を想定していたが、misarta 本体が
> 純関数 API (`fk(model, q) -> Data`) を採用しているため、Python 側もこれに
> 合わせて **fresh Data を返す形** とした。`Data` は `misarta.Data(model)`
> でも生成可能だが、通常は FK が返すものを受け取れば良い。

**公開モジュール/関数 (実装済み)**:
- `misarta.SE3` (class) — `identity`, `from_homogeneous`, `rotation`, `translation`, `homogeneous`, `inverse`, `__mul__`
- `misarta.JointType` — `revolute(axis)`, `prismatic(axis)`, `fixed()`, `free_flyer()` (static factories)
- `misarta.JointModel` — `name`, `parent`, `joint_type`, `placement`
- `misarta.Model` — `name`, `nq`, `nv`, `njoints`, `n_total`, `gravity`, `joint(idx)`, `joint_id(name)`, `link_id(name)`, `joint_names()`, `link_names()`
- `misarta.Data` — `oMi(i)`, `joint_placement(i)`, `J`, `body_velocity(i)`, `body_acceleration(i)`
- `misarta.forward_kinematics(model, q) -> Data`
- `misarta.compute_joint_jacobian(model, q, joint_id, ref_frame=WORLD)` (LOCAL / WORLD 切替対応)
- `misarta.rnea(model, q, v, a)`
- `misarta.crba(model, q)`
- `misarta.aba(model, q, v, tau)`
- `misarta.build_model_from_urdf(urdf_str, root=None)`
- `misarta.load_urdf(path)`
- `misarta.LOCAL`, `misarta.WORLD`, `misarta.LOCAL_WORLD_ALIGNED` (定数)

### Phase 2 — 高次運動学・幾何
- `frames`, `centroidal`, `manifold`, `limits`, `kinematics_utils`, `collision`

### Phase 3 — アプリ層
- `ik`, `optimization (iLQR)`, `reduced`, `constraint`, `mimic`, `regressor`

### Phase 4 — 解析微分
- `rnea_derivatives`, `aba_derivatives` を解析微分 API として露出

---

## 4. データ変換規約

| Rust 型 | Python 型 | 方法 |
|---------|-----------|------|
| `nalgebra::DVector<f64>` | `numpy.ndarray (n,) float64` | `numpy::PyArray1` 経由 |
| `nalgebra::DMatrix<f64>` | `numpy.ndarray (m,n) float64` | `numpy::PyArray2` 経由 (column-major → row-major で転送) |
| `Vector3<f64>` / `Vector6<f64>` | `numpy.ndarray (3,)` / `(6,)` | 固定長 |
| `Matrix3<f64>` | `numpy.ndarray (3,3)` | 〃 |
| `SE3<f64>` | `PySE3` クラス | `.rotation -> (3,3)`, `.translation -> (3,)` |
| `Model<f64>` | `PyModel` (Arc 共有) | 不変。`Data` が `Arc<Model>` を保持 |
| `Data<f64>` | `PyData` | `&mut self` でアルゴリズム実行 |

入力配列の形状検証は Rust 側で行い、不一致時は `ValueError` を送出。

---

## 5. ビルド/配布

### 5.1 Cargo.toml (misarta-py)

```toml
[package]
name = "misarta-py"
version = "0.1.0"
edition = "2024"

[lib]
name = "_misarta"
crate-type = ["cdylib"]

[dependencies]
misarta  = { path = "../misarta" }
pyo3     = { version = "0.22", features = ["extension-module", "abi3-py39"] }
numpy    = "0.22"
nalgebra = "0.34"
```

### 5.2 pyproject.toml

```toml
[build-system]
requires = ["maturin>=1.7,<2"]
build-backend = "maturin"

[project]
name = "misarta"
version = "0.1.0"
requires-python = ">=3.9"
dependencies = ["numpy>=1.21"]

[tool.maturin]
features = ["pyo3/extension-module"]
module-name = "misarta._misarta"
python-source = "python"
```

### 5.3 開発ワークフロー
```bash
cd articara/misarta-py
python -m venv .venv && source .venv/bin/activate
pip install maturin numpy pytest
maturin develop          # ホットリロード
pytest tests/
```

### 5.4 配布 (将来)
- `cibuildwheel + maturin` で Linux/macOS/Windows × x86_64/arm64 のホイール
- abi3 wheel なので Python バージョンごとの再ビルド不要

---

## 6. 検証戦略

### 6.1 単体テスト (Rust 側)
`misarta-py/tests/` に Python の pytest を置く。Rust 単体テストは misarta 本体に既存の 376 件があるため重複させない。

### 6.2 Pinocchio 比較テスト (重要)
同一 URDF (例: `tests/model/urdf/test_robot.urdf`) を `pinocchio` と `misarta` 両方で読み、以下を `numpy.allclose(atol=1e-10)` で比較:
- `forward_kinematics` の `oMi[i]`
- `compute_frame_jacobian` の J 行列
- `rnea(q, v, a)` のトルク
- `crba(q)` の質量行列
- `aba(q, v, tau)` の加速度

これは Pinocchio との互換性 baseline として、リグレッション検出にも使う。

### 6.3 メモリ安全テスト
- `PyData` が dropped した後も `PyModel` が独立して使えることを `weakref` で確認
- `Model` が drop された後の `Data` 操作で SEGV しないこと (Arc により保証されるが明示テスト)

---

## 7. 実行計画 (MVP)

### Step 1 — スカフォールド ✅
- [x] `articara/Cargo.toml` workspace に `misarta-py` を追加 (default-members には含めない)
- [x] `articara/misarta-py/Cargo.toml`, `pyproject.toml` 作成
- [x] `src/lib.rs` で `#[pymodule]` 宣言
- [x] `python/misarta/__init__.py` 作成

### Step 2 — 変換層 ✅
- [x] `src/conv.rs`: numpy ↔ nalgebra ヘルパ (`pyarray_to_vec`, `dvector_to_pyarray`, `dmatrix_to_pyarray`, `pyarray_to_vector3`, `pyarray_to_matrix3`, `vector3_to_pyarray`, `matrix3_to_pyarray`, `check_len`)
- [x] `src/se3.rs`: `PySE3` (rotation, translation, homogeneous, `__mul__`, inverse, identity, from_homogeneous)

### Step 3 — Model/Data ✅
- [x] `src/model.rs`: `PyJointType`, `PyJointModel`, `PyModel` (Arc<Model<f64>> を保持)
- [x] `src/data.rs`: `PyData` (oMi, joint_placement, J, body_velocity, body_acceleration)

### Step 4 — アルゴリズム ✅
- [x] `src/algorithms.rs`: `forward_kinematics`, `compute_joint_jacobian`, `rnea`, `crba`, `aba`
- [x] 入力配列の長さ検証 (`check_len`)

### Step 5 — ローダー ✅
- [x] `src/loaders.rs`: `build_model_from_urdf(urdf_str, root=None)`, `load_urdf(path)`

### Step 6 — テスト ✅
- [x] `tests/test_smoke.py`: import / モジュール属性 / SE3 基本演算 / JointType ファクトリ (5 件)
- [x] `tests/test_fk.py`: URDF 読込 / FK / Jacobian (6 件)
- [x] `tests/test_dynamics.py`: CRBA 対称性 / RNEA / ABA / τ = M·a + h の整合性 (4 件)
- [ ] (Phase 2) `tests/test_pinocchio_parity.py`: pinocchio がある環境での差分比較

### Step 7 — ドキュメント
- [x] `misarta-py/README.md`: ビルド手順 + 簡単な使用例
- [ ] (Phase 2) `python/misarta/_misarta.pyi`: 型スタブ (Phase 1 API 分)

### 完了条件 (MVP) — **全項目達成**
1. ✅ `maturin develop` がエラーなく完了
2. ✅ Python から `import misarta` ができる
3. ✅ 既存 URDF (`test_robot.urdf`) を読み込み、FK/Jac/RNEA/CRBA/ABA が動く
4. ✅ `pytest tests/` が全 pass (15 件)
5. ✅ `aba(model, q, v, rnea(model, q, v, a)) ≈ a` 往復テスト成立 (atol=1e-9)

---

## 8. 命名/慣例

- **PyPI パッケージ名**: `misarta`
- **Python モジュール**: `misarta` (実体は `misarta._misarta` のネイティブ拡張 + Python ラッパ)
- **クラス名**: Pinocchio (Python) と概ね揃える (`Model`, `Data`, `SE3`, `JointModelRevolute` 等)
- **関数名**: snake_case (Python 慣例) に合わせる。Pinocchio Python は `forwardKinematics` 等 camelCase だが、本バインディングでは Python 流儀を優先

---

## 9. 範囲外 (本計画では扱わない)

- 自動微分 (Dual) を Python から透過利用すること → Phase 4 で解析微分 API を露出
- `articara` (GUI/エディタ) の Python 露出
- MuJoCo / robstride 連携
- gRPC / REST API

---

## 10. リスクと対策

| リスク | 対策 |
|--------|------|
| nalgebra が column-major、numpy が row-major デフォルト | `numpy::PyArray2` の `as_array()` でストライドを尊重。テストで shape/値を確認 |
| `Model` の lifetime と Python GC のミスマッチ | `Arc<Model<f64>>` で参照カウント管理。`Data` が Arc を保持 |
| PyO3 0.22 / numpy 0.22 の API 変更 | バージョンを Cargo.toml で固定。PyO3 0.23 への移行は Phase 2 以降で検討 |
| Workspace 内の依存バージョン重複 | `misarta` が `nalgebra 0.34` を要求、`misarta-py` も合わせる |
