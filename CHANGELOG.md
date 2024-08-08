# CHANGELOG

## v0.1.2-rc.2 (2024-08-08)

### Fix

* fix: torch load weights only as security issue ([`6fb7d39`](https://github.com/sachahu1/Diffusion-Model-Framework/commit/6fb7d394440f51e1acb1c90893b75cf2c74d3e1f))

* fix: allow loading BetaScheduler without initialization ([`193bd96`](https://github.com/sachahu1/Diffusion-Model-Framework/commit/193bd969d56129a708ef1d312fbd5430c5df2a30))

* fix: remove unnecessary dependency ([`76b8ab1`](https://github.com/sachahu1/Diffusion-Model-Framework/commit/76b8ab11b9498eee69556326a860a3cfd130d037))

* fix: Release latest on main only ([`13568f2`](https://github.com/sachahu1/Diffusion-Model-Framework/commit/13568f2dbdd33e4aa145f137d3143c79a904e51a))

### Unknown

* Merge pull request #15 from sachahu1/fix/only-release-latest-on-main

fix: run inference from checkpoint ([`e80b03e`](https://github.com/sachahu1/Diffusion-Model-Framework/commit/e80b03e89aa2be7398eda82acc843287ca384953))

## v0.1.2-rc.1 (2024-08-08)

### Fix

* fix: give dispatch permissions to trigger CI ([`5f3333a`](https://github.com/sachahu1/Diffusion-Model-Framework/commit/5f3333a598603ad9ff8abdeea31d77d1d3bf5154))

### Unknown

* Merge pull request #14 from sachahu1/ci/fix-dispatch-permissions

fix: give dispatch permissions to trigger CI ([`1cc96f7`](https://github.com/sachahu1/Diffusion-Model-Framework/commit/1cc96f72700d7b1e6b69744b7fa4a9d10f4908aa))

## v0.1.1 (2024-08-08)

### Unknown

* Merge pull request #13 from sachahu1/dev

Release documentation and fixed CI/CD pipeline ([`4f7fae9`](https://github.com/sachahu1/Diffusion-Model-Framework/commit/4f7fae94864190ee9c6fce9808255bd22ec0369b))

## v0.1.1-rc.4 (2024-08-08)

### Fix

* fix: Mistake in workflow name ([`f80c59b`](https://github.com/sachahu1/Diffusion-Model-Framework/commit/f80c59b11aaa461349650f5bf54972134911c673))

### Unknown

* Merge pull request #12 from sachahu1/fix/rename-workflow

fix: Mistake in workflow name ([`09afacd`](https://github.com/sachahu1/Diffusion-Model-Framework/commit/09afacda6b961489fab7608c71882fc3ce576ce6))

## v0.1.1-rc.3 (2024-08-08)

### Ci

* ci: Use workflow_dispatch to trigger docs ([`83c77e6`](https://github.com/sachahu1/Diffusion-Model-Framework/commit/83c77e6f81b85f6964c7b2fdb0a0547f51ff4f73))

### Fix

* fix: commit forgotten __init__,py ([`c670b4a`](https://github.com/sachahu1/Diffusion-Model-Framework/commit/c670b4a9cf922585776d3eb35ef8c21fd419c4a5))

### Unknown

* Merge pull request #11 from sachahu1/fix/build-docs-not-triggered

Trigger doc release via workflow_dispatch ([`c737587`](https://github.com/sachahu1/Diffusion-Model-Framework/commit/c737587cea206bdde74632db99a85232dacaef13))

## v0.1.1-rc.2 (2024-08-07)

### Fix

* fix: pipeline build and release docs at the tag node ([`e56d111`](https://github.com/sachahu1/Diffusion-Model-Framework/commit/e56d111d65f9fe1a47f70488a779125d42589030))

### Unknown

* Merge pull request #10 from sachahu1/ci/fix-pipeline

fix: pipeline build and release docs at the tag node ([`bf6c44a`](https://github.com/sachahu1/Diffusion-Model-Framework/commit/bf6c44a3e688dd2dd69f98f5918481ccfcbe5523))

## v0.1.1-rc.1 (2024-08-07)

### Documentation

* docs: improved readme ([`ad8a5ff`](https://github.com/sachahu1/Diffusion-Model-Framework/commit/ad8a5ffa9ffb5a5c7b1fc051190e04f663b9bf5e))

* docs: document diffusion trainer ([`5fd5eb7`](https://github.com/sachahu1/Diffusion-Model-Framework/commit/5fd5eb7117f62bb41360ee89f7e0456f0d37f4b9))

* docs: short type hints and version fix ([`5da9b6e`](https://github.com/sachahu1/Diffusion-Model-Framework/commit/5da9b6e5f5bf59c847a6b186b9dbb87483c797da))

* docs: write introduction ([`22c9b8d`](https://github.com/sachahu1/Diffusion-Model-Framework/commit/22c9b8d89e3412e8f49ca3947406d640f0f002bf))

* docs: document rest of the library ([`d00dfb9`](https://github.com/sachahu1/Diffusion-Model-Framework/commit/d00dfb942dce19a7abe986a40350ddf2a2142f7a))

* docs: document GaussianDiffuser ([`d7c5dc1`](https://github.com/sachahu1/Diffusion-Model-Framework/commit/d7c5dc1529c3365659572d3715361b0163c10a2d))

* docs: document BetaSchedulers ([`3c162e6`](https://github.com/sachahu1/Diffusion-Model-Framework/commit/3c162e6fde1db170e5577f546081da2676bc608f))

* docs: document BaseDiffuser ([`edfa807`](https://github.com/sachahu1/Diffusion-Model-Framework/commit/edfa807b47fa84e67337ae202605cd60bf97b55a))

### Fix

* fix: remove workflow_run as it doesn&#39;t work as expected

on push to main or dev ([`b55b29c`](https://github.com/sachahu1/Diffusion-Model-Framework/commit/b55b29ce846342876dcfc7257d01370ef098f80d))

* fix: Wait for release tag to build docs ([`033400f`](https://github.com/sachahu1/Diffusion-Model-Framework/commit/033400fb735d19b27e140b185055a087d9449daa))

### Unknown

* Merge pull request #9 from sachahu1/docs/documentation

Document entire library to date ([`20f072e`](https://github.com/sachahu1/Diffusion-Model-Framework/commit/20f072ee1287194125e9e261c2e499652befd46c))

* lock poetry ([`a842476`](https://github.com/sachahu1/Diffusion-Model-Framework/commit/a842476b270cb67af38df1dd4a0c75da9b9d3226))

## v0.1.0 (2024-08-06)

### Feature

* feat: add some status badges

fix url ([`e87f3ec`](https://github.com/sachahu1/Diffusion-Model-Framework/commit/e87f3ec4be6a1abe1ddd70b4423cf7c42cfcdbf6))

* feat: automatic build and release of docs

not sure where the syntax error is

try pushing docs

remove multiversion ([`09338a5`](https://github.com/sachahu1/Diffusion-Model-Framework/commit/09338a52e2fc8412d9459e7dacbc29d442b59aaa))

* feat: add release process

download workflow artifact

update to cache@v4

unpack into dist folder

add publishing permissions

try on legacy

test with pypa

working pipeline

feat: Publish release

Test releases

test syntax

echo

additional permissions

protect releases on dev

test full version

fix to master

try a new version

only allow releases on main

back to alpha version

fix typo

deactivate other pipelines for dynamic versioning testing

fix workflows

add dynamic versioning to pyproject toml

unshallow fetch

try semantic release

try matching more branches

feat: adjust semantic-release config

feat: bump __init__

trigger pypi publishing on tags

try fixing regex

try another syntax

github regex is weird

publish to pypi

fix: testpypi publishing

move to pypi

adjust pyproject.toml ([`1278e12`](https://github.com/sachahu1/Diffusion-Model-Framework/commit/1278e12942361d33274d55f6e4b6bd5d3f5ead15))

* feat: test pipeline

rename package and add missing test dependencies

add initial test folder

try simplifying pipeline

fix poetry install --with syntax

fix pyproject toml and relock

poetry show with test

did i forget poetry run?

try fixing syntax

additional package caching

forgot to install test dependencies

combine install tests with unit tests ([`f53241f`](https://github.com/sachahu1/Diffusion-Model-Framework/commit/f53241f318f63cbb55398edc1bf9b0e6f2913bd0))

### Unknown

* Merge pull request #4 from sachahu1/dev

Initial Release ([`7d933a9`](https://github.com/sachahu1/Diffusion-Model-Framework/commit/7d933a9f754627631ece6887500ebfea469f10ba))

* Merge pull request #3 from sachahu1/feature/improve_framework

Automated testing, release and docs ([`aed8627`](https://github.com/sachahu1/Diffusion-Model-Framework/commit/aed8627277434c88491dbd6700e50ac0b763f337))

* fix docs deployments

feat: automatic doc on release, formatting, examples

fix: urls

fix: upload url ([`4c48cd4`](https://github.com/sachahu1/Diffusion-Model-Framework/commit/4c48cd4bf294366d25619888e14ffccad6b5a3d3))

* Merge pull request #2 from sachahu1/feature/framework-improvements

Improved Framework ([`2401f12`](https://github.com/sachahu1/Diffusion-Model-Framework/commit/2401f12f3aa581b181b34cb45dc680c15b88d3f1))

* formatting + new beta schedulers ([`61741bc`](https://github.com/sachahu1/Diffusion-Model-Framework/commit/61741bc967822a0c0b9e3151423a478ae6eb2b30))

* reworked framework ([`0dfa220`](https://github.com/sachahu1/Diffusion-Model-Framework/commit/0dfa22032624bd6ea1c7b99a5c79e827a8ac2d7a))

* update dependencies ([`129b52c`](https://github.com/sachahu1/Diffusion-Model-Framework/commit/129b52cec361c19a5a6fa5857d865b2bcb4b642c))

* Merge pull request #1 from sachahu1/feature/initial-work

Initial work on this diffusion repo. ([`e8ceeeb`](https://github.com/sachahu1/Diffusion-Model-Framework/commit/e8ceeeb04972e02e517b7b07c733e9bf5ff195c9))

* format ([`34a705d`](https://github.com/sachahu1/Diffusion-Model-Framework/commit/34a705db8e8f2231193cf58b496e5ee4feebe9e7))

* remove outdated packages ([`c62d882`](https://github.com/sachahu1/Diffusion-Model-Framework/commit/c62d882ec1476c61ea4346197a8d74ac64de3842))

* Add missing pieces ([`ed67d24`](https://github.com/sachahu1/Diffusion-Model-Framework/commit/ed67d24dce533eadfb85af1ff5c322b135f156ad))

* missing __init__ ([`b45f911`](https://github.com/sachahu1/Diffusion-Model-Framework/commit/b45f911fcfa04db6e4480703ed0773f3e0e3ab86))

* syntax fixes and improvements ([`3b6b600`](https://github.com/sachahu1/Diffusion-Model-Framework/commit/3b6b6005450f69cf779c749114c865eff598643b))

* minor correction ([`658c1c9`](https://github.com/sachahu1/Diffusion-Model-Framework/commit/658c1c9f1623120f66ae0f52e89032a7b47fd5b2))

* Initial diffusion code ([`6e43a4e`](https://github.com/sachahu1/Diffusion-Model-Framework/commit/6e43a4edcfdcbf115680f062b61d0723cbcaf2d3))

* Add sampling image ([`f192df3`](https://github.com/sachahu1/Diffusion-Model-Framework/commit/f192df36e1dd90b00c44b2d31b8078874f21e9ce))

* add gitignore ([`e1e83fa`](https://github.com/sachahu1/Diffusion-Model-Framework/commit/e1e83fa13fc21fd135013829c2c5afa983df632a))

* Initial commit ([`229d09e`](https://github.com/sachahu1/Diffusion-Model-Framework/commit/229d09e6738f1ae0a9565876ff75b80e74df2849))
