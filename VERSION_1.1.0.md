# MIRT4FC 1.1.0 版本说明与使用指南

> 本文档对应 **MIRT4FC 1.1.0**（发布日期：2026.09.30），介绍该版本的更新内容、核心功能与完整使用方法。

---

## 1. 版本概览

`MIRT4FC` 是一个基于 **iStEM 算法**（improved Stochastic EM）高效实现多种**迫选（Forced-Choice, FC）**多维 IRT 模型的 R 包。当前支持：

- **MUPP-2PLM**（Multi-Unidimensional Pairwise Preference 2PL Model）
- **MUPP-GGUM** 等迫选展开模型
- 计划持续扩展更多模型

除项目参数估计外，还提供：能力参数估计（MAP / EAP / MLE）、模拟作答矩阵生成、能力与项目参数的标准误（SE）估计，以及一组实证数据集。

### 版本历史

| 版本 | 日期 | 主要变化 |
|------|------|----------|
| 0.0.0 | 2023.10.15 | 初始版本：2PL 模型项目参数估计、模拟数据生成、能力参数估计、实证数据 |
| 1.0.0 | 2025.06.14 | 新增项目参数的标准误（SE）估计功能 |
| **1.1.0** | **2026.09.30** | **新增 iStEM 固定项目参数选项；为 L-BFGS-B 优化器增加稳定性保护；移除 `lvmcomp` 依赖（将 `calcu_sigma_cmle` 移植为纯 R 实现）** |

---

## 2. 1.1.0 核心更新

### 2.1 固定项目参数（Fixed Item Parameters）

iStEM 算法现在支持在估计过程中**固定部分题块的项目参数 $(a, d)$ 不变**，仅更新其余题块的参数。适用于：

- 锚题（anchor items）设计，用于量表链接与等值
- 已有标定参数的量表，仅需估计新增题目
- 需要通过固定参数约束模型识别的场景

新增三个参数：

| 参数 | 类型 | 说明 |
|------|------|------|
| `fixed.blocks` | 数值向量 | 需要固定的题块索引；仅这些块的参数保持固定 |
| `fixed.a` | 矩阵 `(blocksize × length(fixed.blocks))` | 固定块的 $a$ 参数取值；为 `NULL` 时使用内部初值 |
| `fixed.d` | 矩阵 `(blocksize × length(fixed.blocks))` | 固定块的 $d$ 参数取值；仅前 `blocksize-1` 行生效，最后一行自动重算为 $-\sum$ 其余行 |

> **说明**：该选项支持所有 `blocksize`（2/3/4）与所有作答格式（`pick`/`rank`/`mole`）。未指定 `fixed.blocks` 时行为与旧版本完全一致。

### 2.2 L-BFGS-B 优化器稳定性保护

在项目参数估计（Equation 8）的目标函数中加入了**数值溢出保护**：

```r
v <- -1 * P.Yj_2PL_3_rank(aj = x[1:3], dj = c(x[4:5], -sum(x[4:5])), thetaj, yj)
# final guard: on rare numeric overflow/NaN in the Hessian perturbation points,
# return a large finite value so L-BFGS-B never sees fn = Inf/NaN
if (!is.finite(v)) { v <- 1e10 }
```

当目标函数因数值溢出产生 `Inf`/`NaN`（尤其在 Hessian 扰动点计算时）时，返回一个**较大的有限值**而非非有限值，避免 L-BFGS-B 优化器报错中断。该保护覆盖全部 `2PL` 变体（`_2`、`_3_pick`、`_3_rank`、`_4_pick`、`_4_rank`、`_4_mole`）。

### 2.3 移除 `lvmcomp` 依赖（移植 `calcu_sigma_cmle`）

1.1.0 起，`lvmcomp` 不再作为依赖包。为保证功能不受影响，原 `lvmcomp` 中用于估计特质间相关矩阵的 C++ 函数（`src/calcu_sigma_cmle.cpp`）被**逐行移植为纯 R 实现** `R/calcu_sigma_cmle.R`：

| lvmcomp (C++/RcppArmadillo) | MIRT4FC 1.1.0 (纯 R) |
|---|---|
| `obj_func_cpp` | `obj_func` |
| `calcu_sigma_cmle_cpp(theta, tol)` | `calcu_sigma_cmle(theta, tol = 1e-5)` |

移植后 `calcu_sigma_cmle` 已导出（`export(calcu_sigma_cmle)`）；`src/` 不再含该 C++ 文件，DESCRIPTION 也不再引入 `RcppArmadillo`。

**功能**：在 iStEM 的 M 步中，根据采样的潜在特质矩阵 $\theta$ 用**约束极大似然估计（CMLE）**估计特质间相关矩阵 $\Sigma$，即在 $\Sigma \succ 0$、$\mathrm{diag}(\Sigma)=1$ 约束下最小化

$$f(\Sigma) = \mathrm{tr}(\Sigma^{-1}\hat{\Sigma}) + \log\det(\Sigma),\quad \hat{\Sigma} = \frac{\theta^\top \theta}{N}$$

采用**带回溯线搜索的梯度下降**（梯度 $-\Sigma^{-1}\hat{\Sigma}\Sigma^{-1}+\Sigma^{-1}$），每步投影对角线为 1，若目标未下降或非正定则步长减半。当 `fix.sigma = FALSE` 时由各 2PL kernel 调用，失败则回退为 `cor(theta)`。



**对功能与效率的影响**：

| 维度 | 影响 |
|---|---|
| 计算功能 | 基本等价：接口、算法、失败回退逻辑均保留 |
| 性能 | 单次调用 R 版略慢于 C++/Armadillo，但矩阵仅 $D \times D$；且每个 EM 迭代仅调用一次，主瓶颈是逐题块 L-BFGS-B 与逐被试 ARMS 采样，故整体影响轻微 |
| 依赖/构建 | 显著改善：去掉 `lvmcomp` 及 C++/Armadillo 依赖，安装更轻、编译失败风险更低 |

---

## 3. 安装

### 3.1 从 GitHub 安装开发版

```r
install.packages("devtools")
library(devtools)
devtools::install_github("flywithwinds/MIRT4FC")
library(MIRT4FC)
```

### 3.2 依赖要求

- R（>= 3.5.0）
- 必需依赖：`armspp`、`doParallel`、`foreach`、`coda`、`stats`、`utils`、`Matrix`、`parallel`、`methods`、`mvnfast`、`nloptr`
- 编译需求：GNU make（`NeedsCompilation: yes`）
- **（1.1.0 起）不再依赖 `lvmcomp`**，也不再引入 `RcppArmadillo` 的 `LinkingTo` 编译链；原 `lvmcomp` 的 `calcu_sigma_cmle` 已移植为纯 R 实现（见 [2.3](#23-移除-lvmcomp-依赖移植-calcu_sigma_cmle)）

---

## 4. 核心函数一览

| 函数 | 作用 |
|------|------|
| `iStEM()` | 使用 iStEM 算法估计 FC 模型的项目参数（及 SE） |
| `data.sim()` | 根据项目参数与潜在特质生成模拟作答矩阵 |
| `theta.est()` | 估计潜在特质（能力）参数（MAP / EAP / MLE） |
| `calcu_sigma_cmle()` | **（1.1.0 起为纯 R 实现）** 用 CMLE（回溯梯度下降）估计特质间相关矩阵 |

### 4.1 `iStEM()` 主要参数

```r
iStEM(Y, BID, positive = rep(TRUE, nrow(BID)),
      blocksize = 3, res = "rank", M = 10, B = 20, model = "2PL",
      SE = "Louis", sigma = NULL, theta = NULL, fix.sigma = FALSE,
      burnin.maxitr = 40, maxitr = 500, eps1 = 1.5, eps2 = 0.4,
      frac1 = 0.2, frac2 = 0.5, cores = NULL, h = NULL,
      fixed.blocks = NULL, fixed.a = NULL, fixed.d = NULL)
```

| 参数 | 说明 |
|------|------|
| `Y` | 作答矩阵（被试数 × 题块数） |
| `BID` | 题块信息表（题数 × 3），列依次为 `Block`、`Item`、`Dim` |
| `blocksize` | 题块大小，FC(2/3/4) |
| `res` | 作答格式：`pick` / `rank` / `mole` |
| `SE` | 标准误方法：`MCMC`/`FDM`/`XPD`/`CDM`/`RES`/`Louis`/`Sandwich`/`complete` |
| `fix.sigma` | 是否估计 $\sigma$ |
| `maxitr` | 最大迭代次数 |
| `eps1`、`eps2` | 稳定性与收敛准则 |
| `cores` | 并行核心数 |
| `h` | 差分法扰动常数（建议 `1e-5`） |
| `fixed.blocks` / `fixed.a` / `fixed.d` | **（1.1.0 新增）** 固定项目参数相关设置 |

> **作答格式等价关系**：`pick-2` ↔ `rank-2` ↔ `mole-2` 等价；`rank-3` ↔ `mole-3` 等价。

---

## 5. 使用示例

### 示例 1：MUPP-2PL 模型完整模拟流程

模拟 1000 名被试、6 个维度、每维度 10 个题目、共 20 个三元题块，生成作答矩阵并估计项目参数：

```r
library(MIRT4FC)
D <- 6                                   # 维度数
nitem.per.dim <- 10                      # 每维度题目数
nblock <- D * nitem.per.dim / 3          # 题块数
set.seed(123456)

# 模拟题块-题目-维度对应表
BID <- data.frame(
  Block = rep(1:nblock, each = 3),
  Item  = rep(1:3, nblock),
  Dim   = c(combn(D, 3)[, sample(choose(D, 3), nblock, replace = TRUE)])
)

# 模拟项目参数真值
item.par <- data.frame(a = seq_len(D * nitem.per.dim))
item.par <- within(item.par, {
  a <- runif(D * nitem.per.dim, 0.7, 3)
  b <- rnorm(D * nitem.per.dim)
  d <- a * b
})
item.par$d <- c(t(aggregate(item.par$d, by = list(BID$Block),
                            function(x) x - mean(x))[, -1]))

# 模拟潜在特质真值
N <- 1000
v <- matrix(0.5, D, D); diag(v) <- 1
theta <- mvnfast::rmvn(N, seq(0, 0, length.out = D), sigma = v)

# 生成模拟作答矩阵
Y <- data.sim(item.par, theta, BID, blocksize = 3, res = 'rank')

# 估计项目参数
fit <- iStEM(Y, BID, maxitr = 100, blocksize = 3, res = 'rank',
             fix.sigma = TRUE)
print(fit)
```

### 示例 2：实证数据示例（2PL-RANK 模型）

```r
library(MIRT4FC)
Y <- MAP_data                            # 内置实证数据
BID <- data.frame(                       # 题块-题目-维度对应表
  Block = rep(1:88, each = 3),
  Item  = rep(1:3, times = 88),
  Dim   = c(11, 1, 14, 15, 13, 18, 6, 2, 7, 1, 21, 24, 23, 9, 22, 5, 3, 8,
            6, 18, 20, 9, 7, 5, 19, 1, 9, 14, 22, 10, 23, 21, 8, 3, 20, 10,
            22, 3, 17, 4, 23, 24,
            8, 4, 24, 21, 12, 10, 9, 11, 8, 17, 12, 2, 15, 11, 14, 24, 22,
            15, 13, 16, 14, 22, 4, 14, 10, 7, 6, 14, 19, 6, 4, 13, 3, 14,
            15, 2, 8, 3, 11, 18, 23, 20, 24, 15,
            22, 3, 20, 1, 14, 21, 16, 18, 4, 13, 16, 18, 5, 1, 8, 23, 2,
            24, 11, 19, 23, 15, 12, 11, 10, 20, 9, 21, 10, 4, 16, 7, 2, 3,
            12, 16, 10, 6, 13, 16, 21, 16, 17, 20,
            10, 19, 13, 7, 5, 15, 11, 23, 24, 8, 6, 11, 2, 19, 15, 17, 20,
            18, 9, 7, 12, 5, 9, 7, 22, 17, 24, 16, 6, 17, 13, 23, 7, 15,
            17, 5, 8, 19, 14, 18, 3, 12, 22, 4, 5,
            21, 13, 1, 23, 9, 13, 11, 22, 9, 19, 21, 8, 21, 12, 6, 16, 1,
            23, 9, 1, 19, 2, 6, 11, 18, 24, 10, 7, 18, 17, 5, 7, 3, 22, 4,
            2, 8, 20, 17, 15, 8, 3, 24, 12, 10, 22,
            2, 11, 23, 19, 19, 13, 17, 6, 20, 24, 9, 17, 5, 20, 12, 6, 19,
            18, 16, 15, 21, 7, 5, 1, 18, 2, 4, 14, 1, 13, 12, 16, 12, 2,
            20, 4, 5, 10, 4, 1, 21, 14, 3)
)

# 估计项目参数
fit <- iStEM(Y, BID, maxitr = 150, blocksize = 3, res = 'rank',
             fix.sigma = TRUE)
print(fit)
```

### 示例 3：固定项目参数（1.1.0 新功能）

固定前 6 个题块的项目参数为其真值，仅估计其余题块：

```r
library(MIRT4FC)
set.seed(123456)

# --- 与示例 1 相同的数据生成过程 ---
BID <- data.frame(Block = rep(1:nblock, each = 3),
                  Item  = rep(1:3, nblock),
                  Dim   = c(combn(D, 3)[, sample(choose(D, 3), nblock, replace = TRUE)]))
item.par <- data.frame(a = seq_len(D * nitem.per.dim))
item.par <- within(item.par, {
  a <- runif(D * nitem.per.dim, 0.7, 3)
  b <- rnorm(D * nitem.per.dim)
  d <- a * b
})
item.par$d <- c(t(aggregate(item.par$d, by = list(BID$Block),
                            function(x) x - mean(x))[, -1]))
N <- 1000
v <- matrix(0.5, D, D); diag(v) <- 1
theta <- mvnfast::rmvn(N, seq(-1, 1, length.out = D), sigma = v)
Y <- data.sim(item.par, theta, BID, blocksize = 3, res = 'rank')

# --- 固定设置 ---
fixed.blocks <- c(1:6)                   # 固定前 6 个题块
idx <- as.vector(sapply(fixed.blocks, function(j) (3 * (j - 1) + 1):(3 * j)))

fa <- matrix(item.par$a[idx], nrow = 3, ncol = length(fixed.blocks))  # 固定 a
fd <- matrix(item.par$d[idx], nrow = 3, ncol = length(fixed.blocks))  # 固定 d

# --- 估计（固定块参数保持不变，仅更新其余块）---
fit.ext <- iStEM(Y, BID, maxitr = 30, blocksize = 3, res = 'rank',
                 fix.sigma = TRUE,
                 fixed.blocks = fixed.blocks, fixed.a = fa, fixed.d = fd)
print(fit.ext)
```

> **注意**：`fixed.a` / `fixed.d` 为矩阵时必须按**列**对应 `fixed.blocks` 的顺序排列，每列为该题块内各题目（行）的参数值。`fixed.d` 仅前 `blocksize-1` 行生效，最后一行会由约束 $-\sum$ 其余行自动重算。

### 示例 4：能力参数估计

```r
library(MIRT4FC)
# a、d 为估计得到的项目参数矩阵（blocksize × J）
a <- matrix(item.par$a, nrow = 3, ncol = 20)
d <- matrix(item.par$d, nrow = 3, ncol = 20)

theta_est <- theta.est(Y, a, d, BID = BID, sigma = v,
                       prior = TRUE, blocksize = 3,
                       res = 'rank', model = '2PL')
```

---

## 6. 从旧版本迁移

1.1.0 的 API 与旧版本**向后兼容**：

- 不设置 `fixed.blocks` 时，`iStEM()` 行为与 1.0.0 完全一致。
- 仅在需要固定项目参数时新增使用 `fixed.blocks` / `fixed.a` / `fixed.d`。
- 若你的代码显式调用过 `lvmcomp`（尤其是 `lvmcomp::calcu_sigma_cmle`），请改用本包内的 `calcu_sigma_cmle()`（纯 R 实现，接口与语义一致，见 [2.3](#23-移除-lvmcomp-依赖移植-calcu_sigma_cmle)）；同时移除对 `lvmcomp` 及 `RcppArmadillo` 的依赖声明。

---

## 7. 参考

- 模型文献：Brown et al. (2011)；Morillo et al. (2016)；Stark et al. (2005)
- 相关论文：*A 2PLM-RANK Multidimensional Forced-choice Model and its Fast Estimation*
- 完整参考手册：`MIRT4FC.pdf`
