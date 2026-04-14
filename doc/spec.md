# misarta 機能仕様書

**misarta** — 剛体運動学ライブラリ

- **Crate 名**: `misarta`
- **名前の由来**: misa (Misato) + art (Articulation) + ta (Takara)
- **配置**: `articara/misarta/`（独立した Cargo クレート）
- **依存**: `nalgebra 0.34.2`（行列演算）、`num-dual 0.10`（自動微分、dev-dependencies）

---

## 1. 概要

Pinocchio（C++ 剛体力学ライブラリ）と同等の運動学機能を Rust で実装した独立クレート。

---

## 2. 設計原則

### 2.1 参照透明性

すべてのアルゴリズム関数は **純粋関数** として実装されている。入力は不変参照 `&Model` と構成ベクトル `&[f64]` のみで、出力として新しい `Data` を返す。副作用・グローバル状態は一切なく、同一入力に対して常に同一の出力を保証する。

```rust
// 順運動学: (不変モデル, 構成ベクトル) → 計算結果
pub fn forward_kinematics(model: &Model, q: &[f64]) -> Data
```

### 2.2 Model / Data 分離（Pinocchio 哲学）

Pinocchio と同じく、不変のロボット記述（`Model`）と可変の計算結果（`Data`）を構造的に分離する：

- `Model`: ロボットのトポロジ、関節型、固定変位、慣性パラメータ（不変）
- `Data`: 順運動学の配置結果、ヤコビアン等（アルゴリズム呼び出しごとに新規生成）

### 2.3 自動微分対応

`num-dual` クレートの二重数（Dual number）と組み合わせることで、同じ数式から自動微分でヤコビアンを計算できる。テストにおいて、解析的ヤコビアンと自動微分ヤコビアンの一致を検証済み。

```rust
// 同じ関数を f64 でも Dual64 でも評価可能
fn end_effector_2link<D: DualNum<f64> + Copy>(q: &[D; 2]) -> [D; 3] {
    let c0 = q[0].cos();
    // ...
}
```

### 2.4 Rust らしい設計パターン

| パターン | 適用箇所 |
|---------|---------|
| enum + match | 関節型（Revolute / Prismatic / Fixed / FreeFlyer）の分岐 |
| Builder | `ModelBuilder` によるモデル構築 |
| 型エイリアス | `SE3 = Isometry3<f64>`, `Motion = Vector6<f64>` |
| 純粋関数群 | `se3::compose()`, `se3::exp()`, `se3::log()` 等 |
| ジェネリクス | `DualNum<f64>` トレイト境界による自動微分対応 |

---

## 3. ファイル構成

| ファイル | 行数 | 内容 |
|---------|-----|------|
| `src/se3.rs` | 246 | SE(3) Lie 群ユーティリティ（同次変換、exp/log、skew、空間ベクトル変換） |
| `src/joint.rs` | 200 | 関節型 enum（Revolute / Prismatic / Fixed / FreeFlyer）、forward / motion_subspace |
| `src/model.rs` | 204 | ロボットモデル（`Model` + `ModelBuilder`）、リンク慣性、ツリー構造 |
| `src/fk.rs` | 161 | 順運動学（Forward Kinematics）— 純粋関数 |
| `src/jacobian.rs` | 172 | 幾何学的ヤコビアン計算 |
| `src/data.rs` | 35 | 計算結果データ構造体 |
| `src/lib.rs` | 6 | モジュール登録 |
| `tests/kinematics.rs` | 162 | 運動学統合テスト |
| `tests/autodiff.rs` | 137 | 自動微分テスト |
| **合計** | **1323** | |

---

## 4. 仕様詳細

### 4.1 SE(3) Lie 群 (`se3.rs`)

SE(3)（Special Euclidean group）の操作を純粋関数として提供する。

**型定義**

| 型エイリアス | 実体 | 意味 |
|------------|------|------|
| `SE3` | `Isometry3<f64>` | 3D 空間の剛体配置（回転 + 並進） |
| `Motion` | `Vector6<f64>` | 空間速度ベクトル（ツイスト）: [角速度; 並進速度] |
| `Force` | `Vector6<f64>` | 空間力ベクトル（レンチ）: [トルク; 力] |

**構成・変換関数**

| 関数 | 機能 |
|------|------|
| `identity()` | 単位配置 |
| `from_rotation_and_translation(rot, trans)` | 回転行列 + 並進ベクトルから SE(3) 構築 |
| `from_homogeneous(m)` | 4×4 同次行列から SE(3) 構築 |
| `to_homogeneous(se3)` | SE(3) → 4×4 同次行列 |
| `rotation_matrix(se3)` | 回転行列 $R \in \mathbb{R}^{3\times3}$ 抽出 |
| `translation(se3)` | 並進ベクトル $t \in \mathbb{R}^3$ 抽出 |

**合成・逆変換**

| 関数 | 機能 |
|------|------|
| `compose(a, b)` | 配置合成: $a \cdot b$ |
| `inverse(se3)` | 逆配置: $M^{-1}$ |
| `act_on_point(se3, point)` | 点の変換: $R \cdot p + t$ |

**指数写像・対数写像（Lie 代数 ↔ Lie 群）**

| 関数 | 機能 |
|------|------|
| `exp(twist)` | se(3) ツイスト → SE(3) 配置（Rodrigues の公式） |
| `log(se3)` | SE(3) → se(3) ツイスト（対数写像） |

指数写像は Rodrigues の公式に基づき、V 行列を用いて並進部を計算する：

$$V = I + \frac{1 - \cos\theta}{\theta^2} [\omega]_\times + \frac{\theta - \sin\theta}{\theta^3} [\omega]_\times^2$$

$$t = V \cdot v$$

ここで $\omega$ は回転ベクトル、$\theta = \|\omega\|$、$v$ はツイストの並進部。

**空間代数（Featherstone 記法）**

| 関数 | 機能 |
|------|------|
| `skew(v)` | 歪対称行列 $[v]_\times$ |
| `motion_cross_matrix(se3)` | 6×6 空間運動変換行列（随伴表現） |
| `force_cross_matrix(se3)` | 6×6 空間力変換行列（随伴の双対） |

### 4.2 関節型 (`joint.rs`)

Pinocchio 互換の関節型を `enum JointType` で表現する。

| バリアント | DOF (nq / nv) | 構成 | 説明 |
|-----------|---------------|------|------|
| `Revolute { axis }` | 1 / 1 | 角度 $\theta$ | 固定軸まわりの回転 |
| `Prismatic { axis }` | 1 / 1 | 変位 $d$ | 固定軸方向の並進 |
| `Fixed` | 0 / 0 | なし | 剛体結合 |
| `FreeFlyer` | 7 / 6 | $(x,y,z,q_x,q_y,q_z,q_w)$ | 6-DOF 浮動ベース |

**メソッド**

| メソッド | 機能 |
|---------|------|
| `nq()` | 構成空間の次元 |
| `nv()` | 速度空間（接空間）の次元 |
| `forward(q)` | 構成 → 関節配置 $M_J(q) \in SE(3)$ |
| `motion_subspace(q)` | 運動部分空間行列 $S \in \mathbb{R}^{6 \times n_v}$：$v_J = S \dot{q}$ |

**便利コンストラクタ**: `revolute_x()`, `revolute_y()`, `revolute_z()`, `prismatic_x()`, `prismatic_y()`, `prismatic_z()`

### 4.3 モデル (`model.rs`)

**`JointModel` 構造体**

| フィールド | 型 | 説明 |
|-----------|------|------|
| `name` | `String` | 関節名 |
| `joint_type` | `JointType` | 関節型 |
| `parent` | `usize` | 親関節インデックス（0 = universe） |
| `placement` | `SE3` | 親関節フレームからの固定変位 |

**`Model` 構造体（不変）**

| フィールド | 型 | 説明 |
|-----------|------|------|
| `joints` | `Vec<JointModel>` | 全関節（index 0 = universe ダミー） |
| `inertias` | `Vec<LinkInertia>` | リンク慣性 |
| `q_idx` / `v_idx` | `Vec<usize>` | q / v ベクトルへのインデックスマッピング |
| `nq` / `nv` | `usize` | 全構成 / 速度次元 |
| `gravity` | `Vector3<f64>` | ワールド座標系の重力ベクトル |

**`ModelBuilder`**: Builder パターンによるモデル構築。`add_joint()` でチェーン追加し、`build()` で不変 `Model` を生成。

### 4.4 順運動学 (`fk.rs`)

```rust
pub fn forward_kinematics(model: &Model, q: &[f64]) -> Data
```

各関節 $i$ をトポロジカル順序（親→子）で処理：

$$\text{joint\_placements}[i] = \text{placement}_i \cdot M_{J_i}(q_i)$$
$$\text{oMi}[i] = \text{oMi}[\text{parent}(i)] \cdot \text{joint\_placements}[i]$$

結果として `Data.oMi[i]` にワールド座標系での各関節フレーム配置が格納される。

### 4.5 ヤコビアン (`jacobian.rs`)

```rust
pub fn compute_joint_jacobian(model: &Model, q: &[f64], joint_idx: usize) -> DMatrix<f64>
```

ワールドフレームにおける幾何学的ヤコビアン $J \in \mathbb{R}^{6 \times n_v}$ を計算する。

対象関節から根まで遡りながら、各祖先関節の寄与を列に書き込む：

$$J_{\text{angular}} = R_i \cdot s_{\omega,i}$$
$$J_{\text{linear}} = R_i \cdot s_{v,i} + \omega_i \times (p_{\text{target}} - p_i)$$

ここで $R_i$ は関節 $i$ のワールド回転、$s_i$ は運動部分空間ベクトル、$p_i$ は関節 $i$ のワールド位置。

---

## 5. テスト

全 27 テストが通過。

### 5.1 単体テスト（17 件）

| モジュール | テスト名 | 検証内容 |
|-----------|---------|---------|
| `se3` | `identity_compose` | 単位元との合成 |
| | `inverse_roundtrip` | $M \cdot M^{-1} = I$ |
| | `exp_log_roundtrip` | $\log(\exp(\xi)) = \xi$ |
| | `exp_pure_translation` | 純並進の指数写像 |
| | `skew_cross_product` | $[a]_\times b = a \times b$ |
| `joint` | `revolute_z_quarter_turn` | Z 軸 90° 回転 |
| | `prismatic_x_displacement` | X 軸並進 |
| | `fixed_is_identity` | 固定関節 = 単位配置 |
| | `revolute_subspace_is_axis` | 運動部分空間 = 軸ベクトル |
| `model` | `build_simple_chain` | 2 関節チェーン構築 |
| `fk` | `fk_zero_config` | ゼロ構成の FK |
| | `fk_shoulder_90deg` | 肩 90° の FK |
| | `fk_both_90deg` | 肩 + 肘 90° の FK |
| | `fk_is_pure` | 同一入力 → 同一出力（参照透明性） |
| `jacobian` | `jacobian_two_link_zero_config` | ゼロ構成のヤコビアン値 |
| | `jacobian_numerical_validation` | 有限差分との一致 |
| | `jacobian_is_pure` | 同一入力 → 同一出力（参照透明性） |

### 5.2 統合テスト — 運動学（6 件、`tests/kinematics.rs`）

| テスト名 | 検証内容 |
|---------|---------|
| `three_link_fk_straight` | 3 リンク直進構成 |
| `three_link_fk_folded` | 3 リンク屈曲構成 |
| `three_link_full_fold` | 3 リンク完全折り返し（$\pi$ 回転） |
| `jacobian_three_link_finite_diff` | 3 リンク全関節での有限差分検証 |
| `prismatic_plus_revolute` | 直動 + 回転の混合関節 |
| `branched_tree` | 分岐するツリー構造 |

### 5.3 統合テスト — 自動微分（4 件、`tests/autodiff.rs`）

| テスト名 | 検証内容 |
|---------|---------|
| `autodiff_matches_analytical_jacobian` | AD ヤコビアン = 解析的ヤコビアン |
| `autodiff_at_zero_config` | ゼロ構成での AD 微分値の検証 |
| `autodiff_value_matches_f64` | `Dual64` の実数部 = `f64` の値 |
| `autodiff_pure_deterministic` | 同一入力 → 同一ヤコビアン（決定性） |

---

## 6. nalgebra の採用理由

行列演算ライブラリとして **nalgebra** をそのまま採用した。検討した代替案と比較結果：

| ライブラリ | 利点 | 不採用理由 |
|-----------|------|-----------|
| **nalgebra** ✅ | Rust エコシステム標準、`Isometry3` で SE(3) 直接表現、`num-dual` と互換 | — |
| `ndarray` | NumPy ライクな API | 固定サイズ行列なし、幾何学型なし |
| `faer` | 高性能線形代数 | 2025 年時点で幾何学型未対応、エコシステム未成熟 |
| `glam` | ゲーム向け高速 | `f32` 専用、学術精度不足 |

nalgebra は Pinocchio が依存する C++ Eigen と同等の位置づけであり、`Isometry3`, `UnitQuaternion`, `Rotation3` 等の幾何学型を標準提供する点で最適である。

---

## 7. 対応予定（未対応）機能

Pinocchio との機能差分を以下に整理する。すべて対応予定であり、優先度順に記載する。

### 7.1 動力学

| 機能 | Pinocchio 相当 | 説明 | 優先度 |
|------|---------------|------|--------|
| **RNEA**（逆動力学） | `rnea()` | $\tau = M(q)\ddot{q} + C(q,\dot{q})\dot{q} + g(q)$ — 関節トルク計算。$O(n)$ 再帰アルゴリズム | 高 |
| **CRBA**（慣性行列） | `crba()` | 質量行列 $M(q)$ の計算。制御・最適化で必須 | 高 |
| **ABA**（順動力学） | `aba()` | $\ddot{q} = M^{-1}(\tau - C\dot{q} - g)$ — 加速度計算。シミュレーションに必須 | 高 |
| コリオリ行列 | `computeCoriolisMatrix()` | $C(q, \dot{q})$ | 中 |
| 重力項 | `computeGeneralizedGravity()` | $g(q)$ | 中 |
| 非線形効果 | `nonLinearEffects()` | $C\dot{q} + g$ | 中 |

### 7.2 運動学（追加アルゴリズム）

| 機能 | Pinocchio 相当 | 説明 | 優先度 |
|------|---------------|------|--------|
| 逆運動学 (IK) | — | 目標位置/姿勢から関節角を求める反復解法 | 高 |
| フレーム配置 | `updateFramePlacements()` | 関節以外の任意フレーム（エンドエフェクタ、センサ等）の配置 | 中 |
| ボディフレームヤコビアン | `computeJointJacobians()` | ローカルフレーム表現でのヤコビアン | 中 |
| ヤコビアン時間微分 | `computeJointJacobiansTimeVariation()` | $\dot{J}(q, \dot{q})$ | 中 |
| 運動学的ヘッシアン | `computeJointKinematicHessians()` | FK の 2階微分 | 低 |
| FK 微分 | `computeForwardKinematicsDerivatives()` | FK の $q, v, a$ に関する解析的微分 | 低 |

### 7.3 接触・拘束

| 機能 | 説明 | 優先度 |
|------|------|--------|
| 接触動力学 | 接触拘束付きの順/逆動力学 | 中 |
| **閉ループ拘束** | パラレルリンク等の閉リンク機構をツリー構造 + 拘束として扱う（§7.8 参照） | 中 |
| 拘束付き動力学 | 等式/不等式拘束下での動力学計算 | 中 |
| 衝撃動力学 | 衝突時の速度不連続の計算 | 低 |

### 7.4 動力学微分

| 機能 | 説明 | 優先度 |
|------|------|--------|
| RNEA 微分 | $\partial\tau/\partial q$, $\partial\tau/\partial\dot{q}$, $\partial\tau/\partial\ddot{q}$ | 中 |
| ABA 微分 | $\partial\ddot{q}/\partial q$, $\partial\ddot{q}/\partial\dot{q}$, $\partial\ddot{q}/\partial\tau$ | 中 |
| 運動エネルギー微分 | $\partial KE / \partial q$ | 低 |

### 7.5 エネルギー・運動量

| 機能 | Pinocchio 相当 | 説明 | 優先度 |
|------|---------------|------|--------|
| 運動エネルギー | `computeKineticEnergy()` | $\frac{1}{2}\dot{q}^T M \dot{q}$ | 中 |
| ポテンシャルエネルギー | `computePotentialEnergy()` | $-\sum m_i g^T p_i$ | 中 |
| 重心 (CoM) | `centerOfMass()` | ロボット全体の重心位置・速度・加速度 | 高 |
| 重心ヤコビアン | `jacobianCenterOfMass()` | CoM のヤコビアン | 高 |
| セントロイダルモメンタム | `computeCentroidalMomentum()` | 角運動量・線運動量。ヒューマノイド制御に必須 | 高 |

### 7.6 モデル入出力

| 機能 | 説明 | 優先度 |
|------|------|--------|
| URDF パーサー | URDF ファイルからモデル構築（articara の既存パーサーを流用可能） | 高 |
| SDF パーサー | SDF (Gazebo) ファイル対応 | 低 |
| MJCF パーサー | MuJoCo XML 対応 | 低 |
| メッシュ読み込み | コリジョン/ビジュアルジオメトリの読み込み | 低 |

### 7.7 その他

| 機能 | 説明 | 優先度 |
|------|------|--------|
| Lie 群積分 | $q \oplus v\Delta t$（構成空間上の積分、FreeFlyer のクォータニオン積分を含む） | 中 |
| リグレッサー | 慣性パラメータの線形リグレッサー $Y(q,\dot{q},\ddot{q})\pi = \tau$ | 低 |
| コード生成 | シンボリック微分・最適化コード生成（CasADi 相当） | 低 |

### 7.8 閉リンク機構への対応方針

現状 misarta はツリー構造（開リンク）のみに対応している。パラレルリンク機構（4節リンク等）や閉ループ構造を持つロボットは、Pinocchio と同じアプローチで対応する：

**方針: ツリー構造 + 拘束**

1. 閉ループの一辺を仮想的に切断し、ツリー構造として表現する
2. 切断箇所に **ループ閉鎖拘束**（loop closure constraint）を定義する
3. 拘束付き動力学ソルバーで拘束力を計算し、運動方程式に組み込む

```text
物理構造（閉ループ）:          misarta 内部表現（ツリー + 拘束）:

  A ──── B                     A ──── B
  |      |         →           |      |
  C ──── D                     C      D
                                  ↕ 拘束: pos(C) = pos(D)
```

具体的には以下の API を追加予定：

- `ConstraintModel`: 拘束の記述（位置一致、相対配置固定等）
- `ConstraintData`: 拘束のヤコビアン $K$ と残差 $\gamma$
- 拘束付き動力学: $M\ddot{q} + C\dot{q} + g = \tau + K^T\lambda$ と $K\ddot{q} + \dot{K}\dot{q} = 0$ の連立

この方式により、ツリーベースの高効率アルゴリズム（RNEA, ABA）を維持しつつ、閉ループを拘束として付加的に処理できる。
