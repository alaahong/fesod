# RC 发布 Pipeline（GitHub Action）

> 为 Apache Fesod (Incubating) 的每个 RC 轮次自动完成：
> 校验门槛 → 源码包签名 → SVN(dist/dev) 分发 → Nexus staging 部署（不 close）→ Git tag/分支推送。
> 设计文档：`docs/superpowers/specs/2026-08-27-rc-release-github-action-design.md`

## 一、组成

| 文件 | 作用 |
|---|---|
| `.github/workflows/release-candidate.yml` | 编排入口，`workflow_dispatch` 手动触发 |
| `release-tools/ci/env.sh` | 共享环境与派生变量（分支/tag/制品名/临时目录） |
| `release-tools/ci/import-gpg.sh` | 导入签名私钥、配置 gpg-agent batch 模式、写 gpg 口令临时文件 |
| `release-tools/ci/setup-creds.sh` | 运行时生成 `~/.m2/settings.xml`（ASF 凭据，0600，不打印） |
| `release-tools/ci/gate-verify.sh` | 门槛：pom 版本断言 + RAT + 编译 + LICENSE/NOTICE/disclaimer 存在 |
| `release-tools/ci/create-src-package.sh` | `git archive` 源码包 + gpg 签名 + sha512 + **自我校验** + 二进制审计 |
| `release-tools/ci/publish-svn.sh` | 提交源码包+`.asc`+`.sha512` 到 `dist/dev/incubator/fesod/<rev>-rc<rc>/` |
| `release-tools/ci/publish-nexus.sh` | 带签名 `mvn clean deploy` 到 Nexus staging（**不 close**） |
| `release-tools/ci/push-git.sh` | 创建并推送 tag `<rev>-rc<rc>` 与分支 `release-<rev>-RC<rc>` 到 apache/fesod |
| `release-tools/ci/summary.sh` | 汇总输出各发布链接（成功失败都运行） |

## 二、触发与输入

在 GitHub 仓库 **Actions → “RC Release (Candidate)” → Run workflow**：

| 输入 | 必填 | 说明 |
|---|---|---|
| `revision` | 是 | 发布版本，如 `2.1.0-incubating` |
| `rc` | 是 | RC 轮次，如 `1` |
| `base-branch` | 否 | RC 的源码基线分支，默认 `main`（需已包含版本 bump 提交） |

派生关系（`revision=2.1.0-incubating, rc=1`）：
- Git tag：`2.1.0-incubating-rc1`
- 分支：`release-2.1.0-incubating-RC1`
- 源码包：`apache-fesod-2.1.0-incubating-src.tar.gz`（+`.asc`+`.sha512`）
- SVN 目录：`dist/dev/incubator/fesod/2.1.0-incubating-rc1/`

## 三、前置准备（首次使用前）

### 1. 版本 bump（手动，一次）
在 `base-branch` 上提交：`pom.xml` 的 `<revision>` 改为目标发布版本（如 `2.1.0-incubating`）。
工作流会据此断言，不匹配即中断。

### 2. 仓库 Secrets（必需）
工作流从仓库 Secrets 读取，**不落仓库、不落日志**：

| Secret | 用途 |
|---|---|
| `ASF_USERNAME` | ASF LDAP 账号，用于 SVN 与 Nexus |
| `ASF_PASSWORD` | ASF LDAP 密码 |
| `GPG_PRIVATE_KEY` | 签名私钥（armored，或 base64 的 armored） |
| `GPG_PASSPHRASE` | 私钥口令 |

Git 推送使用 `GITHUB_TOKEN`（Actions 自动提供，无须配置）。

> **ASF 官方仓库注意**：`apache/fesod` 的 Secrets 需通过 **ASF Infra（JIRA INFRA-*）** 申请添加。
> 建议先在 fork（如 `alaahong/fesod`）的 Settings → Secrets 配置用于联调，通过后再上官方仓库。

### 3. GPG 密钥
签名人必须是发布经理本人的密钥，且其公钥已追加到：
- https://dist.apache.org/repos/dist/dev/incubator/fesod/KEYS
- https://downloads.apache.org/incubator/fesod/KEYS

将对应私钥（armored）作为 `GPG_PRIVATE_KEY`，口令作为 `GPG_PASSPHRASE`。

## 四、执行流程（单 job，顺序执行）

```
checkout(base-branch) ─► Setup JDK8 ─► Install svn
  ─► import-gpg ─► setup-creds ─► gate-verify  [门槛：任一失败即中断，不再发布]
  ─► create-src-package（签名+哈希+二进制审计）
  ─► publish-svn ─► publish-nexus（不 close）─► push-git
  ─► summary（成败均输出链接）
```

`set -euo pipefail` 保证：校验失败 ⇒ job 立即失败 ⇒ 不执行任何 SVN/Nexus/Git 写操作。

## 五、发布后 RM 手动步骤

1. **Close Nexus staging repo**：登录 https://repository.apache.org（`zhangzhe`），
   Staging Repositories 中选中生成的 `orgapachefesod-XXXX` → **Close**，等待规则校验通过。
2. 更新 VOTE 邮件草稿中的 staging 链接（若 close 后编号不变则无需改）。
3. 参照参考邮件发送 `[VOTE]` 到 `dev@fesod.apache.org`。

## 六、密钥与日志安全说明

- 密钥仅在每次 job 的临时 runner 环境存在，任务结束即销毁。
- 全部经 `env:` 注入，脚本用 `${VAR}` 引用；**不写仓库、不进 checkout**。
- 约定（已内置在脚本规范）：**禁 `set -x`、禁 echo 密钥、禁把密码当命令行参数**、
  临时配置文件 `0600`、**禁 cat 配置文件到日志**，对拼接后的值用 `echo "::add-mask:..."`。
- GitHub 会把与 secret 完全相等的值在日志自动打码为 `***`。
- 上传 artifact 不含任何含密钥的临时文件。

## 七、本地复测

脚本可在本地直接跑（需已设置 `REVISION`/`RC` 及对应 `ASF_*`/`GPG_*` 环境变量）。
仅“发布”类脚本（svn/nexus/push）需要凭据与网络；`gate-verify.sh` 与 `create-src-package.sh`
为只读/本地操作，无需密钥。

命令行示例（本地验证门槛，不发布）：
```bash
export REVISION=2.1.0-incubating RC=1
bash release-tools/ci/gate-verify.sh
bash release-tools/ci/create-src-package.sh
```