# Changelog

## [0.1.13](https://github.com/mgd43b/manicule/compare/v0.1.12...v0.1.13) (2026-08-26)


### Bug Fixes

* 25 correctness defects and 8 performance hotspots found by a full-project review ([#316](https://github.com/mgd43b/manicule/issues/316)) ([8dd6fae](https://github.com/mgd43b/manicule/commit/8dd6faedbaa84d05c7acc9d87967901898962ef2))

## [0.1.12](https://github.com/mgd43b/manicule/compare/v0.1.11...v0.1.12) (2026-08-25)


### Bug Fixes

* harden publication leases and watermark clearing ([#314](https://github.com/mgd43b/manicule/issues/314)) ([a4acdc8](https://github.com/mgd43b/manicule/commit/a4acdc8272d4cfc5871f2f476cfc70b918a0220d))

## [0.1.11](https://github.com/mgd43b/manicule/compare/v0.1.10...v0.1.11) (2026-08-25)


### Features

* **connectors:** add Git-backed site indexing ([#311](https://github.com/mgd43b/manicule/issues/311)) ([44e9438](https://github.com/mgd43b/manicule/commit/44e943877577de23eae00f05e685114c24af7aaf))


### Bug Fixes

* recover snapshot and publication retries ([#313](https://github.com/mgd43b/manicule/issues/313)) ([d3647bf](https://github.com/mgd43b/manicule/commit/d3647bfaec77aee754bcc5e9ecaefc9694925c73))

## [0.1.10](https://github.com/mgd43b/manicule/compare/v0.1.9...v0.1.10) (2026-08-24)


### Performance Improvements

* make rebuild replay fast and restartable ([#309](https://github.com/mgd43b/manicule/issues/309)) ([2f90e5a](https://github.com/mgd43b/manicule/commit/2f90e5ac8dba0f775232bf2554646e5fc5624307))

## [0.1.9](https://github.com/mgd43b/manicule/compare/v0.1.8...v0.1.9) (2026-08-24)


### Performance Improvements

* accelerate rebuild and re-embed replay ([#287](https://github.com/mgd43b/manicule/issues/287)) ([f0de4b0](https://github.com/mgd43b/manicule/commit/f0de4b0bb2751101905e68aebe72ca5fb7fed85c))
* batch re-embedding by chunk budget ([#286](https://github.com/mgd43b/manicule/issues/286)) ([21ef18d](https://github.com/mgd43b/manicule/commit/21ef18d2ccdc40f18e7ba58446e2796b470f7808))
* batch rebuild validation ([#288](https://github.com/mgd43b/manicule/issues/288)) ([e30053d](https://github.com/mgd43b/manicule/commit/e30053d4107493bf57df9899d4569e7c8d1e1abc))
* reuse validation checkpoints after takeover replay ([#289](https://github.com/mgd43b/manicule/issues/289)) ([ac8e821](https://github.com/mgd43b/manicule/commit/ac8e821f099d3bbbbfdf1f0406295386bdaf5dd7))

## [0.1.8](https://github.com/mgd43b/manicule/compare/v0.1.7...v0.1.8) (2026-08-22)


### Bug Fixes

* **connectors:** harden Confluence sync consistency ([#281](https://github.com/mgd43b/manicule/issues/281)) ([01be99f](https://github.com/mgd43b/manicule/commit/01be99fef87db986336d9f3acbdcba1ce6d5fe9e))


### Performance Improvements

* avoid re-planning immutable re-embed snapshots ([#282](https://github.com/mgd43b/manicule/issues/282)) ([e453978](https://github.com/mgd43b/manicule/commit/e45397823736a044194b98aed403e3672e74dcf7))
* reuse verified re-embed snapshots ([#278](https://github.com/mgd43b/manicule/issues/278)) ([6ad6379](https://github.com/mgd43b/manicule/commit/6ad63793c55f7ce6d8854d5134c280761ba9a7bc))
* stream re-embed shadow inspection ([#280](https://github.com/mgd43b/manicule/issues/280)) ([e0071ed](https://github.com/mgd43b/manicule/commit/e0071edbe1c033222e0b741e79c68e59cc23f395))

## [0.1.7](https://github.com/mgd43b/manicule/compare/v0.1.6...v0.1.7) (2026-08-22)


### Performance Improvements

* batch whole-index re-embedding ([#276](https://github.com/mgd43b/manicule/issues/276)) ([cf6a6c8](https://github.com/mgd43b/manicule/commit/cf6a6c8233c0c1f9c44b09191e55b14a63e38f7b))

## [0.1.6](https://github.com/mgd43b/manicule/compare/v0.1.5...v0.1.6) (2026-08-21)


### Bug Fixes

* **ingest:** batch and checkpoint rebuild replay and validation ([#274](https://github.com/mgd43b/manicule/issues/274)) ([68ed328](https://github.com/mgd43b/manicule/commit/68ed3288568985f4086f3c3d1c00c6835a929261))

## [0.1.5](https://github.com/mgd43b/manicule/compare/v0.1.4...v0.1.5) (2026-08-21)


### Features

* **storage:** give every stored vector a versioned checksum over its persisted bytes ([#273](https://github.com/mgd43b/manicule/issues/273)) ([734b5dd](https://github.com/mgd43b/manicule/commit/734b5dd1e1a5768c4c150d98f5be220e73929039))


### Performance Improvements

* **ci:** rebalance the test shards, and fix the writer that made them slow ([#271](https://github.com/mgd43b/manicule/issues/271)) ([a141829](https://github.com/mgd43b/manicule/commit/a141829df21471eec5223db204a4457ade8cbfdf))

## [0.1.4](https://github.com/mgd43b/manicule/compare/v0.1.3...v0.1.4) (2026-08-20)


### Bug Fixes

* **ingest:** renew the rebuild lease while a takeover replays its checkpoint ([#269](https://github.com/mgd43b/manicule/issues/269)) ([63ce9aa](https://github.com/mgd43b/manicule/commit/63ce9aab9f8d9d4d88286bc0dcc561b8a5181336))

## [0.1.3](https://github.com/mgd43b/manicule/compare/v0.1.2...v0.1.3) (2026-08-20)


### Features

* **storage:** execute the two derived-index lifecycles the storage document describes ([#267](https://github.com/mgd43b/manicule/issues/267)) ([d699371](https://github.com/mgd43b/manicule/commit/d699371a3b3b62db3e79c4a8b443e9b881623b14))

## [0.1.2](https://github.com/mgd43b/manicule/compare/v0.1.1...v0.1.2) (2026-08-20)


### Bug Fixes

* **connectors:** converge the Data Center inventory when a deep offset outlives the request timeout ([#260](https://github.com/mgd43b/manicule/issues/260)) ([90341cf](https://github.com/mgd43b/manicule/commit/90341cf16068bf6e781288fdbc12b26d1d372798))
* settle a failed rebuild, refuse a busy writer, and stop pages waiting on the fleet ([#266](https://github.com/mgd43b/manicule/issues/266)) ([08b4ecc](https://github.com/mgd43b/manicule/commit/08b4eccd54a341385ea0425a263d3686448a7447)), closes [#257](https://github.com/mgd43b/manicule/issues/257)

## [0.1.1](https://github.com/mgd43b/manicule/compare/v0.1.0...v0.1.1) (2026-08-19)


### Features

* **cli:** hand the session from your own Chrome to manicule via an extension ([#248](https://github.com/mgd43b/manicule/issues/248)) ([af62803](https://github.com/mgd43b/manicule/commit/af628039299ac6fe2367558235b792730015f0e2))
* **connectors:** make installed Chrome with a dedicated profile a login default ([#246](https://github.com/mgd43b/manicule/issues/246)) ([77922d1](https://github.com/mgd43b/manicule/commit/77922d1ba4cdea0aa128210470919459dadb206b))
* **parsing:** embed what a diagram states, not the syntax that draws it ([#253](https://github.com/mgd43b/manicule/issues/253)) ([219735d](https://github.com/mgd43b/manicule/commit/219735d0908f149aa3448eae87585756d9262ac2))


### Bug Fixes

* bound tokenizer work on large blocks, and keep the acquisition lease alive through synchronous preparation ([#255](https://github.com/mgd43b/manicule/issues/255)) ([dec9ea3](https://github.com/mgd43b/manicule/commit/dec9ea3783eaeca1bd2e7b3bb10391747b7e2951))
* **ingest:** verify the reusable manifest once, not once per worker that asks ([#256](https://github.com/mgd43b/manicule/issues/256)) ([fe1c904](https://github.com/mgd43b/manicule/commit/fe1c90461ec17cc6f2da1b5c96a39377dcddc52c))
* publish each distribution from its own environment ([#243](https://github.com/mgd43b/manicule/issues/243)) ([cca64bb](https://github.com/mgd43b/manicule/commit/cca64bb5bec7bafe9b2dc42a913e6e535946ddbc))
* rewrite README links so they resolve on PyPI ([#247](https://github.com/mgd43b/manicule/issues/247)) ([6ff01ba](https://github.com/mgd43b/manicule/commit/6ff01ba139dad50f9531d8fed14241094c61ea61))

## [0.1.0](https://github.com/mgd43b/manicule/compare/v0.1.0...v0.1.0) (2026-08-19)


### Features

* distribute manicule and manicule-mlx on PyPI ([7af475d](https://github.com/mgd43b/manicule/commit/7af475d21f68b0db07aa580dacaa8cfa9e7360b3))


### Bug Fixes

* fence rebuild evidence verification ([#218](https://github.com/mgd43b/manicule/issues/218)) ([e32ab5c](https://github.com/mgd43b/manicule/commit/e32ab5c0f0c880f86768758521de62fc8a810319))
* harden rebuild evidence publication fence ([#219](https://github.com/mgd43b/manicule/issues/219)) ([c08122d](https://github.com/mgd43b/manicule/commit/c08122d87b636b8bf0364df87006ac0e34690a54))
* make partial rebuild settlement honest ([#217](https://github.com/mgd43b/manicule/issues/217)) ([1716892](https://github.com/mgd43b/manicule/commit/1716892066fa2c591ca8926f11c996e0ee817ee4))
* make reset-index clear workspace identity durably ([#237](https://github.com/mgd43b/manicule/issues/237)) ([87b2c2e](https://github.com/mgd43b/manicule/commit/87b2c2edf2dd4b6fc3ffa449b9647e9063728ebe))
