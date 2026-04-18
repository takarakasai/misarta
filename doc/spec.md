# misarta 機能仕様書

**misarta** — Rust 製剛体力学ライブラリ（Pinocchio 相当）

- **Crate 名**: `misarta`
- **名前の由来**: misa (Misato) + art (Articulation) + ta (Takara)
- **配置**: `articara/misarta/`（独立した Cargo クレート）
- **依存**: `nalgebra 0.34`（行列演算）、`parry3d-f64`（衝突検出）、`num-dual 0.10`（自動微分、dev-dependencies）
- **総ソース行数**: 約 19,500 行（テスト含む）
- **テスト数**: **352 件**（全パス）

---

## 1. 概要

Pinocchio（C++ 剛体力学ライブラリ）と同等の運動学・動力学・最適化機能を Rust で実装した独立クレート。Featherstone の空間代数に基づく O(n) アルゴリズム群を提供する。

---

## 2. 設計原則

### 2.1 参照透明性

すべてのアルゴリズム関数は **純粋関数** として実装されている。入力は不変参照 `&Model<T>` と構成ベクトル `&[T]` のみで、出力として新しい値を返す。副作用・グローバル状態は一切なく、同一入力に対して常に同一の出力を保証する。

```rust
pub fn forward_kinematics<T: RealField>(model: &Model<T>, q: &[T]) -> Data<T>
pub fn rnea<T: RealField>(model: &Model<T>, q: &[T], v: &[T], a: &[T]) -> DVector<T>
pub fn aba<T: RealField>(model: &Model<T>, q: &[T], v: &[T], tau: &[T]) -> DVector<T>
```

### 2.2 Model / Data 分離（Pinocchio 哲学）

Pinocchio と同じく、不変のロボット記述（`Model`）と可変の計算結果（`Data`）を構造的に分離する：

- `Model<T>`: ロボットのトポロジ、関節型、固定変位、慣性パラメータ（不変）
- `Data<T>`: 順運動学の配置結果、ヤコビアン、ボディ速度・加速度等（アルゴリズム呼び出しごとに新規生成）

### 2.3 ジェネリクス・自動微分対応

全アルゴリズムは `T: RealField` でジェネリックに実装。`f64` はもちろん、`num-dual` クレートの二重数（Dual number）による自動微分にも対応。テストにおいて、解析的ヤコビアンと自動微分ヤコビアンの一致を検証済み。

### 2.4 Featherstone 空間代数

空間ベクトルの並びは **Featherstone 記法** `[angular(3); linear(3)]` に統一。

### 2.5 Rust らしい設計パターン

| パターン | 適用箇所 |
|---------|---------|
| enum + match | 関節型（Revolute / Prismatic / Fixed / FreeFlyer）の分岐 |
| Builder | `ModelBuilder` によるモデル構築 |
| 型エイリアス | `SE3<T> = Isometry3<T>`, `Motion<T> = Vector6<T>` |
| 純粋関数群 | `se3::compose()`, `se3::exp()`, `se3::log()` 等 |
| ジェネリクス | `T: RealField` トレイト境界による自動微分・記号微分対応 |

---

## 3. ファイル構成

| ファイル | 行数 | 内容 |
|---------|-----|------|
| `src/se3.rs` | 347 | SE(3) Lie 群ユーティリティ（同次変換、exp/log、skew、空間ベクトル変換） |
| `src/joint.rs` | 228 | 関節型 enum（Revolute / Prismatic / Fixed / FreeFlyer）、forward / motion_subspace |
| `src/model.rs` | 648 | ロボットモデル（`Model<T>` + `ModelBuilder`）、リンク慣性、ツリー構造、`MimicJoint` |
| `src/data.rs` | 44 | 計算結果データ構造体（配置、速度、加速度、ヤコビアン） |
| `src/fk.rs` | 496 | 順運動学（0次・1次速度付き・2次加速度付き） |
| `src/jacobian.rs` | 765 | 幾何学的ヤコビアン（ワールド / ローカル / 相対 / マスク / 時間微分） |
| `src/rnea.rs` | 389 | RNEA（逆動力学）— $\tau = M\ddot{q} + C\dot{q} + g$ |
| `src/crba.rs` | 342 | CRBA（質量行列）— $M(q)$ |
| `src/aba.rs` | 577 | ABA（順動力学）— $\ddot{q} = M^{-1}(\tau - h)$ + $M^{-1}$ 計算 |
| `src/rnea_derivatives.rs` | 703 | RNEA 解析的微分 — $\partial\tau/\partial q$, $\partial\tau/\partial v$, $\partial\tau/\partial a$ |
| `src/aba_derivatives.rs` | 289 | ABA 解析的微分 — $\partial\ddot{q}/\partial q$, $\partial\ddot{q}/\partial v$, $\partial\ddot{q}/\partial\tau$ |
| `src/centroidal.rs` | 548 | 重心・セントロイダルモメンタム（CoM, CMM, $\dot{A}_G$, 運動量変化率） |
| `src/constrained.rs` | 342 | 拘束付き前向き動力学 + 衝撃動力学 |
| `src/frames.rs` | 409 | 操作空間フレーム（任意の名前付きフレーム配置・ヤコビアン） |
| `src/collision.rs` | 832 | 衝突検出（parry3d ベース、ACM、ポテンシャル場） |
| `src/ik.rs` | 946 | 逆運動学ソルバー（位置/姿勢/ポーズ/マルチタスク/干渉回避） |
| `src/manifold.rs` | 296 | 構成空間多様体操作（integrate, difference, interpolate） |
| `src/limits.rs` | 166 | 関節制限（位置/速度/トルク クランプ） |
| `src/optimization.rs` | 1384 | 最適化 API（iLQR ソルバー、離散動力学線形化、2次コスト近似） |
| `src/trajectory.rs` | 319 | 軌道補間（線形、3次 Hermite、5次、B-スプライン） |
| `src/kinematics_utils.rs` | 394 | 運動学ユーティリティ（フレーム間距離、セグメント間最近点等） |
| `src/geometry.rs` | 269 | ジオメトリモデル（Box / Sphere / Cylinder / Capsule / Cone / Mesh） |
| `src/urdf.rs` | 1013 | URDF パーサー / ライター（ジオメトリ・mimic 対応） |
| `src/sdf.rs` | 993 | SDF パーサー / ライター（ジオメトリ対応） |
| `src/mesh.rs` | 735 | メッシュ読み込み（STL、Collada 参照） |
| `src/collada.rs` | 1255 | Collada DAE 読み書き（マテリアル・テクスチャ・サブメッシュ対応） |
| `src/reduced.rs` | 1055 | モデルリダクション（Pinocchio `buildReducedModel` 相当） |
| `src/constraint.rs` | 2280 | 拘束ヤコビアン・等式/不等式拘束付き IK（QP ベース） |
| `src/mimic.rs` | 368 | Mimic（連動）関節ユーティリティ（射影、射影行列、トルク射影） |
| `src/qp.rs` | 1100+ | 密 QP ソルバー（プラグイン可能バックエンド: ActiveSet / Clarabel） |
| `src/utils.rs` | 321 | 数値微分ユーティリティ（ヤコビアン、ヘッシアン） |
| `src/lib.rs` | 31 | モジュール登録（31 モジュール） |
| **合計** | **~19,500** | |

---

## 4. 仕様詳細

### 4.1 SE(3) Lie 群 (`se3.rs`)

SE(3)（Special Euclidean group）の操作を純粋関数として提供する。

**型定義**

| 型エイリアス | 実体 | 意味 |
|------------|------|------|
| `SE3<T>` | `Isometry3<T>` | 3D 空間の剛体配置（回転 + 並進） |
| `Motion<T>` | `Vector6<T>` | 空間速度ベクトル（ツイスト）: [角速度; 並進速度] |
| `Force<T>` | `Vector6<T>` | 空間力ベクトル（レンチ）: [トルク; 力] |

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

$$V = I + \frac{1 - \cos\theta}{\theta^2} [\omega]_\times + \frac{\theta - \sin\theta}{\theta^3} [\omega]_\times^2, \quad t = V \cdot v$$

**空間代数（Featherstone 記法）**

| 関数 | 機能 |
|------|------|
| `skew(v)` | 歪対称行列 $[v]_\times$ |
| `spatial_inertia(mass, com, I)` | 6×6 空間慣性行列 |
| `motion_cross(v)` | 運動クロス積 $v \times$ |
| `force_cross(v)` | 力クロス積 $v \times^*$ |
| `motion_cross_matrix(se3)` | 6×6 随伴行列 |
| `force_cross_matrix(se3)` | 6×6 随伴双対行列 |

### 4.2 関節型 (`joint.rs`)

Pinocchio 互換の関節型を `enum JointType<T>` で表現する。

| バリアント | DOF (nq / nv) | 構成 | 説明 |
|-----------|---------------|------|------|
| `Revolute { axis }` | 1 / 1 | 角度 $\theta$ | 固定軸まわりの回転 |
| `Prismatic { axis }` | 1 / 1 | 変位 $d$ | 固定軸方向の並進 |
| `Fixed` | 0 / 0 | なし | 剛体結合 |
| `FreeFlyer` | 7 / 6 | $(x,y,z,q_x,q_y,q_z,q_w)$ | 6-DOF 浮動ベース |

**メソッド**: `nq()`, `nv()`, `forward(q)`, `motion_subspace(q)`, `approx_eq()`

**便利コンストラクタ**: `revolute_x()`, `revolute_y()`, `revolute_z()`, `prismatic_x()`, `prismatic_y()`, `prismatic_z()`

### 4.3 モデル (`model.rs`)

**`Model<T>` 構造体（不変）**

| フィールド | 型 | 説明 |
|-----------|------|------|
| `name` | `String` | ロボット名 |
| `joints` | `Vec<JointModel<T>>` | 全関節（index 0 = universe ダミー） |
| `inertias` | `Vec<LinkInertia<T>>` | リンク慣性 |
| `link_names` | `Vec<String>` | リンク名 |
| `q_idx` / `v_idx` | `Vec<usize>` | q / v ベクトルへのインデックスマッピング |
| `nq` / `nv` | `usize` | 全構成 / 速度次元 |
| `gravity` | `Vector3<T>` | ワールド座標系の重力ベクトル（デフォルト $[0, 0, -9.81]$） |
| `mimic` | `Vec<MimicJoint<T>>` | Mimic（連動）関節拘束のリスト |

**`MimicJoint<T>` 構造体**

URDF の `<mimic>` タグに対応する連動関節の代数的拘束。slave 関節の構成値は master 関節からアフィン写像で決定される：

$$q_{\text{slave}} = m \cdot q_{\text{master}} + o$$

| フィールド | 型 | 説明 |
|-----------|------|------|
| `slave` | `usize` | 従属関節インデックス（1-based） |
| `master` | `usize` | 主関節インデックス（1-based） |
| `multiplier` | `T` | ギア比 $m$ |
| `offset` | `T` | オフセット $o$ |

**`ModelBuilder`**: Builder パターンによるモデル構築。`add_joint()` でチェーン追加、`add_mimic()` で連動関節拘束追加、`build()` で不変 `Model` を生成。`from_model()` で既存モデルからビルダーを再構築。

### 4.4 計算データ (`data.rs`)

**`Data<T>` 構造体**

| フィールド | 型 | 説明 |
|-----------|------|------|
| `joint_placements` | `Vec<SE3<T>>` | 親フレームからの関節配置 $M_J(q)$ |
| `oMi` | `Vec<SE3<T>>` | ワールドフレームでの関節配置 |
| `J` | `DMatrix<T>` | ボディフレーム・ヤコビアン |
| `v` | `Vec<Vector6<T>>` | ボディフレーム空間速度 |
| `a` | `Vec<Vector6<T>>` | ボディフレーム空間加速度 |

### 4.5 順運動学 (`fk.rs`)

| 関数 | 説明 |
|------|------|
| `forward_kinematics(model, q)` | 0 次 FK — 各関節のワールド配置 |
| `forward_kinematics_velocity(model, q, v)` | 1 次 FK — 配置 + ボディ速度 |
| `forward_kinematics_acceleration(model, q, v, a)` | 2 次 FK — 配置 + 速度 + 加速度（Featherstone トリックで重力含む） |

各関節 $i$ をトポロジカル順序（親→子）で処理：

$$\text{oMi}[i] = \text{oMi}[\lambda(i)] \cdot \text{placement}_i \cdot M_{J_i}(q_i)$$

2 次 FK での加速度計算は Featherstone トリックを使用（$a_0 = -g$）し、コリオリ項 $v_i \times v_{J_i}$ を含む。

### 4.6 ヤコビアン (`jacobian.rs`)

| 関数 | 説明 |
|------|------|
| `compute_joint_jacobian` | ワールドフレーム幾何学的ヤコビアン $J \in \mathbb{R}^{6 \times n_v}$ |
| `compute_joint_jacobian_local` | ボディ（ローカル）フレーム・ヤコビアン — $R_i^T$ で回転 |
| `compute_joint_jacobian_time_derivative` | ヤコビアン時間微分 $\dot{J}(q, \dot{q})$ — 中心差分 |
| `compute_relative_jacobian` | 2 フレーム間の相対ヤコビアン |
| `compute_masked_jacobian` | 関節マスク付きヤコビアン（指定関節のみ有効化） |
| `compute_relative_masked_jacobian` | 相対 + マスク付きヤコビアン |

各関数には `_from_data` バリアントがあり、事前計算された `Data` を再利用可能。

### 4.7 逆動力学 — RNEA (`rnea.rs`)

```rust
pub fn rnea<T: RealField>(model: &Model<T>, q: &[T], v: &[T], a: &[T]) -> DVector<T>
```

O(n) 再帰 Newton-Euler アルゴリズム。

| 関数 | 説明 |
|------|------|
| `rnea(model, q, v, a)` | $\tau = M(q)\ddot{q} + C(q,\dot{q})\dot{q} + g(q)$ |
| `compute_gravity(model, q)` | 重力項 $g(q)$ = `rnea(model, q, 0, 0)` |
| `nonlinear_effects(model, q, v)` | 非線形効果 $C\dot{q} + g$ = `rnea(model, q, v, 0)` |

### 4.8 質量行列 — CRBA (`crba.rs`)

```rust
pub fn crba<T: RealField>(model: &Model<T>, q: &[T]) -> DMatrix<T>
```

複合剛体アルゴリズムにより対称正定値質量行列 $M(q) \in \mathbb{R}^{n_v \times n_v}$ を計算。

### 4.9 順動力学 — ABA (`aba.rs`)

| 関数 | 説明 |
|------|------|
| `aba(model, q, v, tau)` | O(n) Articulated Body Algorithm: $\ddot{q} = M^{-1}(\tau - h)$ |
| `compute_minv_times_vec(model, q, tau)` | $M^{-1} \tau$ — ABA で O(n) 計算 |
| `compute_minv(model, q)` | 逆質量行列 $M^{-1}$ 全体 — 列ごとに ABA 適用 |

### 4.10 RNEA 解析的微分 (`rnea_derivatives.rs`)

```rust
pub fn compute_rnea_derivatives<T: RealField>(
    model: &Model<T>, q: &[T], v: &[T], a: &[T]
) -> RneaDerivatives<T>
```

Carpentier & Mansard (RSS 2018) に基づく解析的微分。3 パス構造:

1. **Pass 1** (順方向): 標準 RNEA + 中間量保存 ($v_{pa}, a_{pa}, v_J, Y_i$)
2. **Pass 2** (逆方向): 複合慣性 $Y_c$ 蓄積 + 力の後退蓄積
3. **Pass 3** (列ごとの微分): 各ジョイント $k$ について:
   - 摂動 $\delta v_k, \delta a_k$ を計算
   - サブツリーに順方向伝播 → $\delta f_j$ 計算
   - サブツリー内で $\delta f$ を後退蓄積
   - 変換微分項 $X_k^* [S_k]^{\times *} f_k$ を追加（$\partial/\partial q$ のみ）
   - 祖先へレンチ伝播

**出力**:

| フィールド | 意味 |
|-----------|------|
| `dtau_dq` | $\partial\tau/\partial q$ — $n_v \times n_v$ |
| `dtau_dv` | $\partial\tau/\partial\dot{q}$ — $n_v \times n_v$ |
| `dtau_da` | $\partial\tau/\partial\ddot{q} = M(q)$ — $n_v \times n_v$ |

### 4.11 ABA 解析的微分 (`aba_derivatives.rs`)

```rust
pub fn compute_aba_derivatives(
    model: &Model<f64>, q: &[f64], v: &[f64], tau: &[f64]
) -> AbaDerivatives<f64>
```

間接法（暗黙関数定理）による ABA 微分:

$$\frac{\partial\ddot{q}}{\partial q} = -M^{-1} \frac{\partial\tau}{\partial q}\bigg|_{a=\ddot{q}}, \quad \frac{\partial\ddot{q}}{\partial\dot{q}} = -M^{-1} \frac{\partial\tau}{\partial\dot{q}}\bigg|_{a=\ddot{q}}, \quad \frac{\partial\ddot{q}}{\partial\tau} = M^{-1}$$

手順: ABA → RNEA 微分 → Cholesky 分解 → 行列積。

**出力**:

| フィールド | 意味 |
|-----------|------|
| `ddq_dq` | $\partial\ddot{q}/\partial q$ — $n_v \times n_v$ |
| `ddq_dv` | $\partial\ddot{q}/\partial\dot{q}$ — $n_v \times n_v$ |
| `ddq_dtau` | $\partial\ddot{q}/\partial\tau = M^{-1}$ — $n_v \times n_v$ |

### 4.12 拘束付き動力学 (`constrained.rs`)

| 関数 | 説明 |
|------|------|
| `constrained_forward_dynamics(model, q, v, tau, Jc, γ)` | KKT 系による拘束付き順動力学 |
| `impact_dynamics(model, q, v_pre, Jc, e)` | Newton 反発則による衝撃動力学 |

**拘束付き順動力学** — KKT 系を直接解く:

$$\begin{bmatrix} M & J_c^T \\ J_c & 0 \end{bmatrix} \begin{bmatrix} \ddot{q} \\ \lambda \end{bmatrix} = \begin{bmatrix} \tau - h \\ -\gamma \end{bmatrix}$$

**衝撃動力学** — Newton の反発則 $J_c v^+ = -e \cdot J_c v^-$:

$$\begin{bmatrix} M & J_c^T \\ J_c & 0 \end{bmatrix} \begin{bmatrix} v^+ \\ \Lambda \end{bmatrix} = \begin{bmatrix} M v^- \\ -e \cdot J_c v^- \end{bmatrix}$$

### 4.13 重心・セントロイダル (`centroidal.rs`)

| 関数 | 説明 |
|------|------|
| `total_mass(model)` | ロボット総質量 |
| `compute_com(model, q)` | 重心位置 |
| `compute_com_velocity(model, q, v)` | 重心速度 |
| `compute_com_jacobian(model, q)` | 重心ヤコビアン $J_{com} \in \mathbb{R}^{3 \times n_v}$ |
| `compute_centroidal_momentum_matrix(model, q)` | セントロイダルモメンタム行列 $A_G \in \mathbb{R}^{6 \times n_v}$ |
| `compute_momentum(model, q, v)` | セントロイダルモメンタム $h_G = A_G \dot{q}$ |
| `compute_centroidal_inertia(model, q)` | セントロイダル慣性 $I_G \in \mathbb{R}^{6 \times 6}$ |
| `compute_centroidal_momentum_matrix_time_derivative(model, q, v)` | $\dot{A}_G$ — 中心差分 |
| `compute_momentum_rate(model, q, v, a)` | 運動量変化率 $\dot{h}_G = A_G \ddot{q} + \dot{A}_G \dot{q}$ |

### 4.14 フレーム (`frames.rs`)

任意の名前付きフレーム（エンドエフェクタ、センサ、ツール先端等）を管理。

| 関数 | 説明 |
|------|------|
| `compute_frame_placement(model, q, frame)` | フレームのワールド配置 |
| `compute_all_frame_placements(model, q, frame_model)` | 全フレームの一括計算 |
| `compute_frame_jacobian(model, q, frame)` | フレームのヤコビアン |
| `frames_from_links(model)` | リンク名からフレームモデル自動生成 |

### 4.15 逆運動学 (`ik.rs`)

ダンプド最小二乗法（DLS）ベースの反復 IK ソルバー。

| 関数 | 説明 |
|------|------|
| `solve_joint_position_ik` | 位置 IK |
| `solve_joint_orientation_ik` | 姿勢 IK |
| `solve_joint_pose_ik` | 位置 + 姿勢 IK |
| `solve_frame_pose_ik` | 任意フレームの IK |
| `solve_joint_position_orientation_ik` | 重み付き位置・姿勢 IK |
| `solve_joint_position_ik_with_posture` | ポスチャタスク付き IK |
| `solve_two_task_position_ik` | 2 タスク IK（主タスク + 副タスク） |
| `solve_joint_position_ik_with_collision_avoidance` | 干渉回避 IK |

**設定**:
- `IkConfig`: 最大反復回数、収束トルランス、ステップサイズ、ダンピング（固定 / 適応的マニピュラビリティ）、関節制限
- `Damping::AdaptiveManipulability`: マニピュラビリティ指標に基づくダンピング自動調整

### 4.16 衝突検出 (`collision.rs`)

`parry3d-f64` ベースの衝突検出。

| 関数 | 説明 |
|------|------|
| `collision_pairs[_acm]` | 干渉ペアの検出 |
| `has_collision[_acm]` | 干渉の有無判定 |
| `minimum_distance[_acm]` | 最小距離計算 |
| `collision_potential` | 滑らかなポテンシャル場 $\sum \max(0, d_s - d_{ij})^2$ |
| `collision_potential_gradient` | ポテンシャル勾配（数値微分） |
| `AllowedCollisionMatrix` | 隣接リンク等の衝突無視設定 |

### 4.17 多様体操作 (`manifold.rs`)

FreeFlyer のクォータニオンを考慮した構成空間上の操作。

| 関数 | 説明 |
|------|------|
| `normalize_configuration(model, q)` | クォータニオン正規化 |
| `integrate(model, q, v, dt)` | $q \oplus v\Delta t$（Lie 群積分） |
| `difference(model, q0, q1)` | $q_1 \ominus q_0$（Lie 群差分） |
| `interpolate(model, q0, q1, α)` | 構成空間上の補間 |

### 4.18 関節制限 (`limits.rs`)

| 関数 | 説明 |
|------|------|
| `clamp_configuration(model, q, limits)` | 位置制限クランプ |
| `saturate_velocity(model, v, limits)` | 速度制限飽和 |
| `project_torques(model, tau, limits)` | トルク制限射影 |
| `is_within_configuration_limits(model, q, limits)` | 制限内判定 |

### 4.19 最適化 (`optimization.rs`)

#### iLQR ソルバー

```rust
pub fn solve_ilqr(
    model, q0, v0, u_init,
    q_ref_seq, v_ref_seq, u_ref_seq,
    Q, R, Qf, dt, eps, config
) -> IlqrResult
```

反復線形2次レギュレータ（iLQR）:
- **Backward pass**: Riccati 再帰（$V_x, V_{xx}, K, d$）
- **Forward pass**: ライン探索（Armijo 型）
- **入力拘束**: $u_{\min}, u_{\max}$ によるクランプ
- **状態拘束**: `joint_limits` による構成・速度の射影

| 構造体 | 説明 |
|--------|------|
| `IlqrConfig` | ソルバー設定（反復数、正則化、入力/状態拘束） |
| `IlqrResult` | 結果（最適制御列、軌道、コスト、収束フラグ） |

#### 離散動力学

| 関数 | 説明 |
|------|------|
| `discrete_dynamics_step(model, q, v, tau, dt)` | 半陰的 Euler 積分 |
| `linearize_discrete_dynamics(model, q, v, tau, dt, eps)` | 離散動力学の線形化 $(A, B)$ |

#### コスト近似

| 関数 | 説明 |
|------|------|
| `quadratic_stage_cost_approximation` | ステージコストの 2 次近似 $(l_x, l_u, l_{xx}, l_{uu}, l_{ux})$ |
| `quadratic_terminal_cost_approximation` | 終端コストの 2 次近似 $(l_x, l_{xx})$ |
| `state_error_tangent` | 接空間上の状態誤差 $e = [q \ominus q_{ref}; v - v_{ref}]$ |

### 4.20 軌道補間 (`trajectory.rs`)

| 関数 | 説明 |
|------|------|
| `linear_interpolate` | 線形補間 |
| `cubic_hermite` / `cubic_hermite_derivative` | 3 次 Hermite（位置 + 速度境界条件） |
| `quintic_interpolate` / `quintic_derivative` | 5 次多項式（加速度ゼロ境界条件） |
| `bspline_linear` / `bspline_quadratic` / `bspline_cubic` | B-スプライン（1 次 / 2 次 / 3 次） |

### 4.21 運動学ユーティリティ (`kinematics_utils.rs`)

| 関数 | 説明 |
|------|------|
| `frame_to_frame_jacobian` | 2 フレーム間のヤコビアン |
| `frame_to_frame_placement` | 2 フレーム間の相対配置 |
| `frame_distance` | フレーム間距離 |
| `point_to_plane_distance` | 点と平面の距離 |
| `closest_point_on_segment` | 線分上の最近点 |
| `closest_points_between_segments` | 2 線分間の最近点ペア |
| `point_to_aabb_distance` | 点と AABB の距離 |

### 4.22 ジオメトリモデル (`geometry.rs`)

| 形状 | パラメータ |
|------|-----------|
| `Box` | `x, y, z` |
| `Sphere` | `radius` |
| `Cylinder` | `radius, length` |
| `Capsule` | `radius, length` |
| `Cone` | `radius, length` |
| `Mesh` | `filename, scale` |

`GeometryModel` — ビジュアル/コリジョン形状のコレクション。関節との紐付け・配置を管理。

### 4.23 モデル入出力

**URDF** (`urdf.rs`):
| 関数 | 説明 |
|------|------|
| `load_urdf` / `load_urdf_string` | ファイル / 文字列から `Model` ロード（`<mimic>` タグ対応） |
| `load_urdf_geometry` / `load_urdf_geometry_string` | `Model` + `GeometryModel`（ビジュアル + コリジョン）ロード |
| `write_urdf` / `write_urdf_string` | `Model` → URDF 出力（mimic タグ出力対応） |
| `write_urdf_geometry_string` | ジオメトリ付き URDF 出力 |

URDF の `<mimic joint="master" multiplier="m" offset="o"/>` 要素を自動的にパースし、`Model.mimic` に `MimicJoint` として格納する。ライターも `Model.mimic` から `<mimic>` タグを生成する。

**SDF** (`sdf.rs`):
| 関数 | 説明 |
|------|------|
| `load_sdf` / `load_sdf_string` | ファイル / 文字列から `Model` ロード |
| `load_sdf_geometry` / `load_sdf_geometry_string` | `Model` + `GeometryModel` ロード |
| `write_sdf` / `write_sdf_string` | `Model` → SDF 出力 |
| `write_sdf_geometry_string` | ジオメトリ付き SDF 出力 |

### 4.24 メッシュ (`mesh.rs`)

STL バイナリ/ASCII メッシュの読み込み。Collada DAE への参照パスサポート。

| 関数 | 説明 |
|------|------|
| `load_stl` | STL ファイルからメッシュ読み込み |
| `Mesh` | 頂点・面・法線データの構造体 |

### 4.25 Collada DAE (`collada.rs`)

Collada 1.4.1 フォーマットの読み書き。

| 関数 | 説明 |
|------|------|
| `read_collada` / `read_collada_string` | Collada ファイル / 文字列からメッシュ読み込み |
| `write_collada` / `write_collada_string` | メッシュ→ Collada 出力 |

対応: マテリアル（diffuse / specular）、テクスチャ参照、サブメッシュ、`<polylist>` / `<triangles>`。

### 4.26 モデルリダクション (`reduced.rs`)

Pinocchio の `buildReducedModel` に相当する機能。指定した関節を固定値でロックし、自由度を削減した小さなモデルを生成する。

| 関数 | 説明 |
|------|------|
| `build_reduced_model(model, joints, q_lock)` | 指定関節をロックした縮退モデル生成 |
| `build_reduced_model_with_geometry(model, vis, col, joints, q_lock)` | ジオメトリ付き縮退モデル |
| `reduce_frame_model(frame_model, model, joints, q_lock)` | フレームモデルの縮退 |

**アルゴリズム**:
1. ロック関節の $M_J(q_{\text{lock}})$ を子の配置に吸収
2. ロックされた関節の慣性を **平行軸の定理** で親に統合
3. インデックスのリマッピング（`old_to_new`、`unlocked_ancestor`）
4. ジオメトリの `parent_joint` と `placement` のリマッピング

**検証内容**: FK 一致性（ゼロ/非ゼロ/連続ロック）、慣性マージ、総質量保存、重力トルク一致性、質量行列次元、ABA 一致性、FreeFlyer ロック、パニック条件

### 4.27 拘束ヤコビアン・拘束付き IK (`constraint.rs`)

Pinocchio 互換の拘束ヤコビアンフレームワーク。ループ閉鎖、クロスブランチ IK、相対姿勢拘束を提供する。

**型**:

| 型 | 説明 |
|------|------|
| `ConstraintType` | `Contact6D`（6行）/ `Contact3D`（3行） |
| `ReferenceFrame` | `World` / `Local`（frame1 ローカル） |
| `RigidConstraint<T>` | 2 フレーム間の剛体拘束 |
| `ConstraintModel<T>` | 拘束のコレクション |
| `ConstrainedIkConfig` | DLS ベース拘束 IK 設定 |
| `QpIkConfig` | QP ベース不等式拘束 IK 設定 |
| `ConstrainedIkResult` | ソルバー結果（q, 反復数, 誤差, 収束フラグ） |

**拘束誤差**:

$$e_{6D} = \log\bigl(M_1^{-1}\, M_2\, M_{\text{des}}^{-1}\bigr), \quad e_{3D} = p_2 - (p_1 + R_1\, t_{\text{des}})$$

**拘束ヤコビアン**:

$$J_c = J_2 - J_1$$

| 関数 | 説明 |
|------|------|
| `compute_constraint_error` | 拘束誤差ベクトルの計算 |
| `compute_constraint_jacobian` | 拘束ヤコビアンの計算 |
| `solve_constrained_ik` | 拘束のみ IK（DLS） |
| `solve_task_with_constraints` | 位置タスク + 等式拘束 IK（拡張ヤコビアン） |
| `solve_frame_task_with_constraints` | 6D フレームタスク + 等式拘束 IK |

**QP ベース不等式拘束 IK**:

等式拘束と不等式拘束（関節リミット、ステップ制限）を同時に扱う IK ソルバー。各反復で以下の QP を解く:

$$\min_{dq} \lVert J_t\, dq - e_t\rVert^2 + w^2 \lVert J_c\, dq + e_c\rVert^2 + \lambda^2 \lVert dq\rVert^2 \quad \text{s.t.} \quad A_{iq}\, dq \le b_{iq}$$

| 関数 | 説明 |
|------|------|
| `build_joint_limit_inequalities` | 関節リミット→不等式行列 |
| `build_max_step_inequalities` | ステップ制限→不等式行列 |
| `stack_inequalities` | 不等式のスタック結合 |
| `solve_constrained_ik_qp` | 拘束のみ IK（QP ベース） |
| `solve_task_with_constraints_qp` | 位置タスク + 等式 + 不等式 IK |
| `solve_frame_task_with_constraints_qp` | 6D フレームタスク + 等式 + 不等式 IK |

### 4.28 QP ソルバー (`qp.rs`)

プラグイン可能なバックエンドを備えた密 QP ソルバー。

$$\min_x \frac{1}{2} x^T H x + c^T x \quad \text{s.t.} \quad A_{eq}\, x = b_{eq}, \quad A_{iq}\, x \le b_{iq}$$

**バックエンド選択 (`QpSolver` enum)**:

| バリアント | アルゴリズム | Cargo feature | 外部依存 |
|-----------|-------------|---------------|---------|
| `ActiveSet` *(default)* | プライマル・アクティブセット法（密、自己完結） | なし（常に利用可能） | なし |
| `Clarabel` | 内点コーニックソルバー | `clarabel` | `clarabel 0.11` |

`QpSolver` enum は `#[non_exhaustive]` であり、将来のバリアント追加（例: `Osqp`, `ProxQP`）を破壊的変更なく行える。

**使用方法**:

```rust
use misarta::qp::{solve_qp, QpConfig, QpSolver};

// デフォルト（ActiveSet）
let sol = solve_qp(&h, &c, None, None, Some(&a_iq), Some(&b_iq), None, &QpConfig::default());

// Clarabel を使用（要 `clarabel` feature）
let cfg = QpConfig { solver: QpSolver::Clarabel, ..Default::default() };
let sol = solve_qp(&h, &c, None, None, Some(&a_iq), Some(&b_iq), None, &cfg);
```

**拘束付き IK との連携**: `QpIkConfig.qp_solver` フィールドで IK 内部の QP バックエンドを選択可能。

| 関数 / 型 | 説明 |
|------|------|
| `solve_qp(H, c, A_eq, b_eq, A_iq, b_iq, x0, config)` | 密 QP ソルバー（バックエンド自動ディスパッチ） |
| `QpSolver` | バックエンド選択 enum（`ActiveSet` / `Clarabel`） |
| `QpConfig` | 設定（バックエンド、最大反復、実行可能性トルランス、最適性トルランス） |
| `QpSolution` | 解（x, 目的関数値, ラグランジュ乗数, ステータス） |
| `QpStatus` | `Optimal` / `MaxIterations` / `Infeasible` / `NumericalFailure` |

**ActiveSet アルゴリズム**:
1. Hessian の Cholesky 分解（正則化フォールバック付き）
2. 初期実行可能点: 等式制約の最小ノルム解 + null-space 射影で不等式実行可能化
3. アクティブセット反復:
   - Schur complement ($S = \hat{A}\, H^{-1}\, \hat{A}^T$) による KKT 系の効率的な解法
   - ステップ長計算 + ブロッキング制約の検出
   - ラグランジュ乗数検査による非アクティブ制約の除去

**Clarabel バックエンド**:
- 密行列を CSC 形式に変換し Clarabel の内点ソルバーに委譲
- 等式拘束: `ZeroConeT`、不等式拘束: `NonnegativeConeT`
- `clarabel` Cargo feature が無効時に `QpSolver::Clarabel` を使用するとパニック

**新バックエンドの追加手順**:
1. `QpSolver` enum に新バリアントを追加
2. `solve_qp()` の `match` に対応アームを追加
3. 外部依存の場合は optional dependency + feature flag を追加

### 4.29 数値微分ユーティリティ (`utils.rs`)

| 関数 | 説明 |
|------|------|
| `numerical_jacobian` | 前進差分ヤコビアン |
| `numerical_jacobian_central` | 中心差分ヤコビアン |
| `numerical_jacobian_fk` | FK 専用数値ヤコビアン |
| `numerical_gradient` | 数値勾配 |
| `numerical_hessian` | 数値ヘッシアン |

### 4.30 Mimic（連動）関節 (`mimic.rs`)

URDF の `<mimic>` タグに対応する連動関節サポート。slave 関節の構成値を master 関節からアフィン写像で決定する。

**設計方針**: `JointType` に新バリアントを追加する方式ではなく、**`Model` にサイドバンド情報として持たせる** 方式を採用。これにより FK / RNEA / ABA / CRBA などの既存アルゴリズムは一切変更不要で、q/v ベクトルの**前処理（射影）**のみで対応する。

$$q_{\text{slave}} = m \cdot q_{\text{master}} + o, \quad \dot{q}_{\text{slave}} = m \cdot \dot{q}_{\text{master}}$$

| 関数 | 説明 |
|------|------|
| `enforce_mimic(model, q)` | q ベクトルの slave 値を master から計算し上書き |
| `enforce_mimic_velocity(model, v)` | v ベクトルの slave 値を $m \cdot v_{\text{master}}$ で上書き |
| `independent_q_indices(model)` | 独立（非 slave）q インデックス一覧 |
| `independent_v_indices(model)` | 独立（非 slave）v インデックス一覧 |
| `num_independent_v(model)` | 独立 DOF 数 |
| `mimic_projection_matrix(model)` | 射影行列 $G \in \mathbb{R}^{n_v \times n_{\text{indep}}}$ |
| `expand_independent_velocity(model, v_indep)` | 独立速度→全速度展開 |
| `project_torque(model, tau)` | 全トルク→独立空間射影 $G^\top \tau$ |

**射影行列 $G$**: 独立関節速度を全関節速度にマッピングする行列。

$$\dot{q} = G\, \dot{q}_{\text{indep}}, \quad J_{\text{reduced}} = J \cdot G$$

master が独立、slave の行は `multiplier` 倍の master 列に非零が入る。例（3 関節、j3 が j1 の mimic で multiplier=2）:

$$G = \begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 2 & 0 \end{bmatrix}$$

**ワークフロー**:

```text
q_independent ──► enforce_mimic(model, q) ──► q_full ──► FK / RNEA / ABA
```

IK / 最適化では独立変数のみ扱い、FK 呼び出し前に `enforce_mimic` で展開する。

---

## 5. テスト

全 **352 テスト** が通過（0 失敗）。

| スイート | 件数 |
|----------|------|
| ユニットテスト（`src/`） | 308 |
| 自動微分テスト（`tests/autodiff.rs`） | 4 |
| 運動学統合テスト（`tests/kinematics.rs`） | 6 |
| ローダーテスト（`tests/regression.rs`） | 23 |
| Doctest | 11 |

### モジュール別テスト内訳

| モジュール | 件数 | 主な検証内容 |
|-----------|------|-------------|
| `se3` | 5 | exp/log 往復、合成、逆変換 |
| `joint` | 4 | 各関節型の forward / motion_subspace |
| `model` | 10 | ModelBuilder、チェーン構築、approx_eq（mimic 含む） |
| `fk` | 9 | 0 次 / 1 次 / 2 次 FK、参照透明性 |
| `jacobian` | 17 | ワールド / ローカル / 相対 / マスク / 時間微分、FD 検証 |
| `rnea` | 6 | 重力トルク、非線形効果、加速度の線形性 |
| `crba` | 5 | 対称性、正定値性、RNEA 一致性 |
| `aba` | 7 | CRBA/RNEA 一致性、自由落下、$M^{-1}$ 対称性 |
| `rnea_derivatives` | 10 | $\partial\tau/\partial a = M$ 一致、$\partial\tau/\partial q$ / $\partial\tau/\partial v$ の FD 検証 (1/2/3 リンク) |
| `aba_derivatives` | 8 | $\partial\ddot{q}/\partial\tau = M^{-1}$ 一致、$\partial\ddot{q}/\partial q$ / $\partial\ddot{q}/\partial v$ の FD 検証 |
| `constrained` | 6 | 拘束付き動力学 KKT、衝撃動力学 |
| `centroidal` | 12 | CoM、CMM、セントロイダル慣性、$\dot{A}_G$、運動量変化率 |
| `frames` | 7 | フレーム配置、ヤコビアン |
| `collision` | 11 | 干渉検出、ACM、ポテンシャル場 |
| `ik` | 10 | 位置 / 姿勢 / ポーズ / マルチタスク / 干渉回避 IK |
| `manifold` | 8 | integrate / difference / interpolate |
| `limits` | 3 | クランプ、飽和、射影 |
| `optimization` | 11 | iLQR、離散動力学線形化、コスト近似 |
| `trajectory` | 8 | Hermite / 5 次 / B-スプライン |
| `kinematics_utils` | 10 | フレーム間距離、最近点 |
| `geometry` | 4 | ジオメトリモデル構築 |
| `mesh` | 12 | STL メッシュ読み込み |
| `collada` | 7 | Collada DAE 読み書き |
| `urdf` | 15 | URDF 読み書き、ジオメトリ、mimic |
| `sdf` | 14 | SDF 読み書き、ジオメトリ |
| `reduced` | 27 | モデルリダクション（FK 一致性、慣性マージ、総質量保存、ABA 一致等） |
| `constraint` | 34 | 拘束ヤコビアン、DLS/QP 拘束 IK、不等式拘束（関節リミット、ステップ制限） |
| `qp` | 15 | QP ソルバー（無制約、等式、不等式、ボックス、混合、乗数検証） |
| `mimic` | 8 | enforce_mimic、速度射影、射影行列、トルク射影、FK 連携 |
| `utils` | 5 | 数値微分 |

---

## 6. nalgebra の採用理由

行列演算ライブラリとして **nalgebra** を採用。

| ライブラリ | 利点 | 不採用理由 |
|-----------|------|-----------|
| **nalgebra** ✅ | Rust エコシステム標準、`Isometry3` で SE(3) 直接表現、`num-dual` と互換 | — |
| `ndarray` | NumPy ライクな API | 固定サイズ行列なし、幾何学型なし |
| `faer` | 高性能線形代数 | 幾何学型未対応、エコシステム未成熟 |
| `glam` | ゲーム向け高速 | `f32` 専用、学術精度不足 |

---

## 7. Pinocchio 対応表

| 機能カテゴリ | Pinocchio 関数 | misarta 対応 | 状態 |
|-------------|---------------|-------------|------|
| **順運動学** | `forwardKinematics` (0次) | `forward_kinematics` | ✅ |
| | `forwardKinematics` (1次: 速度) | `forward_kinematics_velocity` | ✅ |
| | `forwardKinematics` (2次: 加速度) | `forward_kinematics_acceleration` | ✅ |
| **ヤコビアン** | `computeJointJacobians` (World) | `compute_joint_jacobian` | ✅ |
| | `computeJointJacobians` (Local) | `compute_joint_jacobian_local` | ✅ |
| | `computeJointJacobiansTimeVariation` | `compute_joint_jacobian_time_derivative` | ✅ |
| **逆動力学** | `rnea` | `rnea` | ✅ |
| | `computeGeneralizedGravity` | `compute_gravity` | ✅ |
| | `nonLinearEffects` | `nonlinear_effects` | ✅ |
| **質量行列** | `crba` | `crba` | ✅ |
| **順動力学** | `aba` | `aba` | ✅ |
| | `computeMinverse` | `compute_minv` / `compute_minv_times_vec` | ✅ |
| **動力学微分** | `computeRNEADerivatives` | `compute_rnea_derivatives` | ✅ |
| | `computeABADerivatives` | `compute_aba_derivatives` | ✅ |
| **拘束動力学** | `constraintDynamics` | `constrained_forward_dynamics` | ✅ |
| | `impulseDynamics` | `impact_dynamics` | ✅ |
| **重心** | `centerOfMass` | `compute_com` / `compute_com_velocity` | ✅ |
| | `jacobianCenterOfMass` | `compute_com_jacobian` | ✅ |
| | `computeCentroidalMomentum` | `compute_momentum` | ✅ |
| | `ccrba` (CMM) | `compute_centroidal_momentum_matrix` | ✅ |
| | `dccrba` ($\dot{A}_G$) | `compute_centroidal_momentum_matrix_time_derivative` | ✅ |
| **フレーム** | `updateFramePlacements` | `compute_frame_placement` | ✅ |
| | `computeFrameJacobian` | `compute_frame_jacobian` | ✅ |
| **IK** | — | `solve_joint_*_ik` (7 種) | ✅ |
| **拘束ヤコビアン** | `computeConstraintJacobian` | `compute_constraint_jacobian` | ✅ |
| **拘束 IK** | — | `solve_constrained_ik`, `solve_task_with_constraints` 等 | ✅ |
| **不等式拘束 IK** | — (QP ベース) | `solve_*_qp` (3 種) | ✅ |
| **モデルリダクション** | `buildReducedModel` | `build_reduced_model` / `_with_geometry` | ✅ |
| **衝突** | (FCL/HPP-FCL) | `collision_pairs`, `minimum_distance` | ✅ |
| **多様体** | `integrate` / `difference` | `integrate` / `difference` | ✅ |
| **URDF** | `buildModelFromUrdf` | `load_urdf` / `load_urdf_geometry` | ✅ |
| **URDF mimic** | URDF `<mimic>` タグ | `MimicJoint` + `enforce_mimic` | ✅ |
| **SDF** | `buildModelFromSdf` | `load_sdf` / `load_sdf_geometry` | ✅ |
| **iLQR** | — (Crocoddyl) | `solve_ilqr` | ✅ |

---

## 8. 対応予定（未対応）機能

### 8.1 動力学

| 機能 | 説明 | 優先度 |
|------|------|--------|
| コリオリ行列 | $C(q, \dot{q})$ を明示的な行列として計算 | 中 |
| 運動エネルギー微分 | $\partial KE / \partial q$ | 低 |

### 8.2 動力学微分（直接法）

| 機能 | 説明 | 優先度 |
|------|------|--------|
| ABA 直接微分 | 間接法 $O(n^3)$ → 直接法 $O(n^2)$ への最適化 | 中 |
| FreeFlyer 用 $\partial S/\partial q$ | FreeFlyer の運動部分空間の $q$ 依存性 | 中 |

### 8.3 関節型

| 機能 | 説明 | 優先度 |
|------|------|--------|
| **Spherical** ジョイント | 3-DOF 球面関節（クォータニオン / 指数座標） | 中 |
| **Planar** ジョイント | 2-DOF 平面関節 | 低 |
| **Universal** ジョイント | 2-DOF ユニバーサル | 低 |
| SDF mimic | SDF ローダでの mimic/transmission 対応 | 低 |

### 8.4 その他

| 機能 | 説明 | 優先度 |
|------|------|--------|
| MJCF パーサー | MuJoCo XML 対応 | 低 |
| メッシュ衝突 | TriMesh 形状の衝突検出 | 低 |
| リグレッサー | 慣性パラメータの線形リグレッサー $Y\pi = \tau$ | 低 |
| CasADi 相当 | シンボリック微分・コード生成 | 低 |

### 8.5 閉リンク機構への対応方針

現状 misarta はツリー構造（開リンク）のみに対応している。パラレルリンク機構（4 節リンク等）は、Pinocchio と同じアプローチで対応する：

**方針: ツリー構造 + 拘束**

1. 閉ループの一辺を仮想的に切断し、ツリー構造として表現する
2. 切断箇所に **ループ閉鎖拘束**（loop closure constraint）を `constraint.rs` の `RigidConstraint` で定義する
3. `compute_constraint_jacobian` で拘束ヤコビアンを計算し、`constrained_forward_dynamics` に渡す
4. IK は `solve_constrained_ik` / `solve_constrained_ik_qp` でループ閉鎖を保ちながら解ける
