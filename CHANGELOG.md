# CHANGELOG


## v1.0.2-rc.1 (2024-11-11)

### Performance Improvements

- Remove unused matplotlib dependency
  ([`119df2a`](https://github.com/sachahu1/Diffusion-Model-Framework/commit/119df2a30e8c1f8af001eb556a046c14dd812916))


## v1.0.1 (2024-11-11)


## v1.0.1-rc.1 (2024-11-11)

### Bug Fixes

- Make release
  ([`d3e493b`](https://github.com/sachahu1/Diffusion-Model-Framework/commit/d3e493bf34c59e81b3abcd23772fa4ed7004c493))

### Chores

- Remove optional jupyter dependency
  ([`843f94b`](https://github.com/sachahu1/Diffusion-Model-Framework/commit/843f94b3825216a68f560677c0014ddfcd70871c))

- Fix documentation
  ([`dd53920`](https://github.com/sachahu1/Diffusion-Model-Framework/commit/dd539204501556bb7a23bb8a4c6165f36b718c9e))


## v1.0.0 (2024-11-05)

### Testing

- Add testing for checkpoints
  ([`d9b29dd`](https://github.com/sachahu1/Diffusion-Model-Framework/commit/d9b29dd13f8476cc1b85979270839bf452f3d5ef))

- Add some initial tests
  ([`5bd3d09`](https://github.com/sachahu1/Diffusion-Model-Framework/commit/5bd3d09219098341fd242af2277450bbfd02f0a7))


## v1.0.0-rc.2 (2024-11-05)

### Documentation

- Fix inference example
  ([`a0f59e9`](https://github.com/sachahu1/Diffusion-Model-Framework/commit/a0f59e949d4f97369b90d46d2d3b44b4ac410e3d))

### Features

- Improve inference framework to support DDIM
  ([`06c4ac1`](https://github.com/sachahu1/Diffusion-Model-Framework/commit/06c4ac1513528c36c639d6f936ba61c060034a19))


## v1.0.0-rc.1 (2024-11-04)

### Bug Fixes

- Minor type fixes
  ([`4e211cb`](https://github.com/sachahu1/Diffusion-Model-Framework/commit/4e211cb77eec4f113f16747665f078f9e760448c))

### Chores

- Support python 3.12.*
  ([`3099ab5`](https://github.com/sachahu1/Diffusion-Model-Framework/commit/3099ab5e865437da4d4936f2b19ed6b72e8b936f))

- Support python3.12
  ([`d9ddb02`](https://github.com/sachahu1/Diffusion-Model-Framework/commit/d9ddb024bcedb7df326fea7765b68ed57db69545))

- Bumping numpy
  ([`4e21dde`](https://github.com/sachahu1/Diffusion-Model-Framework/commit/4e21ddee348a2bbcc1d58e9c8fce680dff504a10))

BREAKING CHANGE: Numpy bumped > 2.0

### Continuous Integration

- Remove python3.9 and add 3.12
  ([`24422be`](https://github.com/sachahu1/Diffusion-Model-Framework/commit/24422be40ff2f885f9dbcc8b67afa339c33d5fba))

BREAKING CHANGE: no longer supports python 3.9

### Features

- Adding DDIM denoising
  ([`2b5595b`](https://github.com/sachahu1/Diffusion-Model-Framework/commit/2b5595b9716de1d4cd412e1229ea49bfaf8b32b0))


## v0.1.2-rc.3 (2024-08-08)

### Bug Fixes

- Map_location is needed for cpu
  ([`c918d3b`](https://github.com/sachahu1/Diffusion-Model-Framework/commit/c918d3b37de7d09f1627ff8f4c395edf055d4e7e))


## v0.1.2-rc.2 (2024-08-08)

### Bug Fixes

- Torch load weights only as security issue
  ([`6fb7d39`](https://github.com/sachahu1/Diffusion-Model-Framework/commit/6fb7d394440f51e1acb1c90893b75cf2c74d3e1f))

- Allow loading BetaScheduler without initialization
  ([`193bd96`](https://github.com/sachahu1/Diffusion-Model-Framework/commit/193bd969d56129a708ef1d312fbd5430c5df2a30))

- Remove unnecessary dependency
  ([`76b8ab1`](https://github.com/sachahu1/Diffusion-Model-Framework/commit/76b8ab11b9498eee69556326a860a3cfd130d037))

- Release latest on main only
  ([`13568f2`](https://github.com/sachahu1/Diffusion-Model-Framework/commit/13568f2dbdd33e4aa145f137d3143c79a904e51a))


## v0.1.2-rc.1 (2024-08-08)

### Bug Fixes

- Give dispatch permissions to trigger CI
  ([`5f3333a`](https://github.com/sachahu1/Diffusion-Model-Framework/commit/5f3333a598603ad9ff8abdeea31d77d1d3bf5154))


## v0.1.1 (2024-08-08)


## v0.1.1-rc.4 (2024-08-08)

### Bug Fixes

- Mistake in workflow name
  ([`f80c59b`](https://github.com/sachahu1/Diffusion-Model-Framework/commit/f80c59b11aaa461349650f5bf54972134911c673))


## v0.1.1-rc.3 (2024-08-08)

### Bug Fixes

- Commit forgotten __init__,py
  ([`c670b4a`](https://github.com/sachahu1/Diffusion-Model-Framework/commit/c670b4a9cf922585776d3eb35ef8c21fd419c4a5))

### Continuous Integration

- Use workflow_dispatch to trigger docs
  ([`83c77e6`](https://github.com/sachahu1/Diffusion-Model-Framework/commit/83c77e6f81b85f6964c7b2fdb0a0547f51ff4f73))


## v0.1.1-rc.2 (2024-08-07)

### Bug Fixes

- Pipeline build and release docs at the tag node
  ([`e56d111`](https://github.com/sachahu1/Diffusion-Model-Framework/commit/e56d111d65f9fe1a47f70488a779125d42589030))


## v0.1.1-rc.1 (2024-08-07)

### Bug Fixes

- Remove workflow_run as it doesn't work as expected
  ([`b55b29c`](https://github.com/sachahu1/Diffusion-Model-Framework/commit/b55b29ce846342876dcfc7257d01370ef098f80d))

on push to main or dev

- Wait for release tag to build docs
  ([`033400f`](https://github.com/sachahu1/Diffusion-Model-Framework/commit/033400fb735d19b27e140b185055a087d9449daa))

### Documentation

- Improved readme
  ([`ad8a5ff`](https://github.com/sachahu1/Diffusion-Model-Framework/commit/ad8a5ffa9ffb5a5c7b1fc051190e04f663b9bf5e))

- Document diffusion trainer
  ([`5fd5eb7`](https://github.com/sachahu1/Diffusion-Model-Framework/commit/5fd5eb7117f62bb41360ee89f7e0456f0d37f4b9))

- Short type hints and version fix
  ([`5da9b6e`](https://github.com/sachahu1/Diffusion-Model-Framework/commit/5da9b6e5f5bf59c847a6b186b9dbb87483c797da))

- Write introduction
  ([`22c9b8d`](https://github.com/sachahu1/Diffusion-Model-Framework/commit/22c9b8d89e3412e8f49ca3947406d640f0f002bf))

- Document rest of the library
  ([`d00dfb9`](https://github.com/sachahu1/Diffusion-Model-Framework/commit/d00dfb942dce19a7abe986a40350ddf2a2142f7a))

- Document GaussianDiffuser
  ([`d7c5dc1`](https://github.com/sachahu1/Diffusion-Model-Framework/commit/d7c5dc1529c3365659572d3715361b0163c10a2d))

- Document BetaSchedulers
  ([`3c162e6`](https://github.com/sachahu1/Diffusion-Model-Framework/commit/3c162e6fde1db170e5577f546081da2676bc608f))

- Document BaseDiffuser
  ([`edfa807`](https://github.com/sachahu1/Diffusion-Model-Framework/commit/edfa807b47fa84e67337ae202605cd60bf97b55a))


## v0.1.0 (2024-08-06)

### Features

- Add some status badges
  ([`e87f3ec`](https://github.com/sachahu1/Diffusion-Model-Framework/commit/e87f3ec4be6a1abe1ddd70b4423cf7c42cfcdbf6))

fix url

- Automatic build and release of docs
  ([`09338a5`](https://github.com/sachahu1/Diffusion-Model-Framework/commit/09338a52e2fc8412d9459e7dacbc29d442b59aaa))

not sure where the syntax error is

try pushing docs

remove multiversion

- Add release process
  ([`1278e12`](https://github.com/sachahu1/Diffusion-Model-Framework/commit/1278e12942361d33274d55f6e4b6bd5d3f5ead15))

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

adjust pyproject.toml

- Test pipeline
  ([`f53241f`](https://github.com/sachahu1/Diffusion-Model-Framework/commit/f53241f318f63cbb55398edc1bf9b0e6f2913bd0))

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

combine install tests with unit tests
