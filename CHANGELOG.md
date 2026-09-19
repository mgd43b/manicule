# Changelog

## [0.2.8](https://github.com/mgd43b/manicule/compare/v0.2.7...v0.2.8) (2026-09-19)


### Features

* **ingest:** re-embed every indexed document under an unchanged fingerprint ([#398](https://github.com/mgd43b/manicule/issues/398)) ([05783ed](https://github.com/mgd43b/manicule/commit/05783ed186e824ffdb2e57cbfef9e448bc294b89))

## [0.2.7](https://github.com/mgd43b/manicule/compare/v0.2.6...v0.2.7) (2026-09-16)


### Features

* **storage:** configure the Qdrant collection's shape, and apply it to collections that exist ([#396](https://github.com/mgd43b/manicule/issues/396)) ([b0ca437](https://github.com/mgd43b/manicule/commit/b0ca4375e37a3142fc9e50fb2a046a639e5636cc))

## [0.2.6](https://github.com/mgd43b/manicule/compare/v0.2.5...v0.2.6) (2026-09-16)


### Features

* persist HTTP and MCP request logs ([#389](https://github.com/mgd43b/manicule/issues/389)) ([66e8371](https://github.com/mgd43b/manicule/commit/66e8371b8c4f7de33740ee06f1d91f1df2abdad0))

## [0.2.5](https://github.com/mgd43b/manicule/compare/v0.2.4...v0.2.5) (2026-09-16)


### Features

* **storage:** carry an existing vector index to another backend without re-embedding ([#387](https://github.com/mgd43b/manicule/issues/387)) ([3dbef8c](https://github.com/mgd43b/manicule/commit/3dbef8cb3caccc2aa73838503afd821fc5510595))

## [0.2.4](https://github.com/mgd43b/manicule/compare/v0.2.3...v0.2.4) (2026-09-15)


### Features

* **collections:** report a corpus no collection holds ([#382](https://github.com/mgd43b/manicule/issues/382)) ([572f3eb](https://github.com/mgd43b/manicule/commit/572f3eb32b54145e77e668ea4a972d8a45fc50f4))
* **connectors:** give the filesystem connector an exclude, as git-site has ([#383](https://github.com/mgd43b/manicule/issues/383)) ([3bd76db](https://github.com/mgd43b/manicule/commit/3bd76dbd80abfb353402688cae19f3e485e37290))


### Bug Fixes

* **rebuild:** stop an incremental snapshot retiring documents it never named ([#384](https://github.com/mgd43b/manicule/issues/384)) ([7489400](https://github.com/mgd43b/manicule/commit/74894000d1cf9dc322b5f089ce7253ee99e88f4d))
* **storage:** reset a derived index on every configured vector backend ([#381](https://github.com/mgd43b/manicule/issues/381)) ([93745d8](https://github.com/mgd43b/manicule/commit/93745d88557357675c7ec5e29f8c83c0c3b539aa))

## [0.2.3](https://github.com/mgd43b/manicule/compare/v0.2.2...v0.2.3) (2026-09-15)


### Bug Fixes

* **bind:** stop the bind policy refusing commands that never bind ([#375](https://github.com/mgd43b/manicule/issues/375)) ([3032412](https://github.com/mgd43b/manicule/commit/3032412c0f50ee1ca63c98fa6fbe869f01e5f453))

## [0.2.2](https://github.com/mgd43b/manicule/compare/v0.2.1...v0.2.2) (2026-09-15)


### Bug Fixes

* **packaging:** pin manicule in every sibling package, and catch a missing pin ([#373](https://github.com/mgd43b/manicule/issues/373)) ([21b112f](https://github.com/mgd43b/manicule/commit/21b112fbb5115df27623f78d970a971d8fefb128))

## [0.2.1](https://github.com/mgd43b/manicule/compare/v0.2.0...v0.2.1) (2026-09-15)


### Bug Fixes

* **plugins:** admit manicule 0.2, which every plugin refused ([#370](https://github.com/mgd43b/manicule/issues/370)) ([14a82d5](https://github.com/mgd43b/manicule/commit/14a82d5ac6b48dc2ee512d0bd51ac0b7f720cc1a))

## [0.2.0](https://github.com/mgd43b/manicule/compare/v0.1.23...v0.2.0) (2026-09-15)


### ⚠ BREAKING CHANGES

* **serve:** --no-authentication serves authoring, which is what it is for ([#366](https://github.com/mgd43b/manicule/issues/366))

### Features

* **collections:** a rule can name a directory, so a synced file joins it ([#365](https://github.com/mgd43b/manicule/issues/365)) ([8b10bfe](https://github.com/mgd43b/manicule/commit/8b10bfeddc615a53747cb60910f18a346fc9346e))
* **serve:** --no-authentication serves authoring, which is what it is for ([#366](https://github.com/mgd43b/manicule/issues/366)) ([ddfcb85](https://github.com/mgd43b/manicule/commit/ddfcb85f958b9c94eb4ecc1c47c973cc1405ba80))


### Bug Fixes

* **release:** stop re-pulling 1.7 GB per release — the Xet log, and the venv's timestamped bytecode ([#364](https://github.com/mgd43b/manicule/issues/364)) ([a117b6b](https://github.com/mgd43b/manicule/commit/a117b6b8329e94ee4a66c8c81693ac550180ceba))
* **tests:** make a leaked connection name itself, and stop a busy writer costing a lease ([#368](https://github.com/mgd43b/manicule/issues/368)) ([a563f0d](https://github.com/mgd43b/manicule/commit/a563f0d3acadc003e63f610c0717abe779d588d5))

## [0.1.23](https://github.com/mgd43b/manicule/compare/v0.1.22...v0.1.23) (2026-09-14)


### Bug Fixes

* **serve:** let --no-authentication reach the preflight that refuses first ([#361](https://github.com/mgd43b/manicule/issues/361)) ([5caf16b](https://github.com/mgd43b/manicule/commit/5caf16b50a53a4129313a3906f01f885d0425330))

## [0.1.22](https://github.com/mgd43b/manicule/compare/v0.1.21...v0.1.22) (2026-09-14)


### Bug Fixes

* keep backend-agnostic paths free of LanceDB so Qdrant runs without AVX2 ([#358](https://github.com/mgd43b/manicule/issues/358)) ([36c2dc9](https://github.com/mgd43b/manicule/commit/36c2dc985fc32f5917448ae64b948ba85138b8b3))
* **release:** make the image's layers reproducible so releases share the model layer ([#356](https://github.com/mgd43b/manicule/issues/356)) ([2912b88](https://github.com/mgd43b/manicule/commit/2912b88084918591010fe766d70f9fde6468dc7f))
* **runtime:** require the configured backend, not just the protocol, before Lance-specific work ([#359](https://github.com/mgd43b/manicule/issues/359)) ([bbb0df8](https://github.com/mgd43b/manicule/commit/bbb0df85d07299036520e79faccfc8f93c16b2d0))

## [0.1.21](https://github.com/mgd43b/manicule/compare/v0.1.20...v0.1.21) (2026-09-14)


### Features

* **embedding:** core-owned query/document prefixes, and an ollama backend usable in the container ([#354](https://github.com/mgd43b/manicule/issues/354)) ([eed3337](https://github.com/mgd43b/manicule/commit/eed33378c8e4c19d75ce1c6f2dcbbbd07c40e6cc))

## [0.1.20](https://github.com/mgd43b/manicule/compare/v0.1.19...v0.1.20) (2026-09-14)


### Bug Fixes

* **packaging:** ship manicule-ollama in the image and publish it to PyPI ([#352](https://github.com/mgd43b/manicule/issues/352)) ([6030415](https://github.com/mgd43b/manicule/commit/60304151e1468a8debf47f1996c753563338e201))

## [0.1.19](https://github.com/mgd43b/manicule/compare/v0.1.18...v0.1.19) (2026-09-14)


### Performance Improvements

* **ci:** run each test shard in parallel, and read the pins rather than the prose ([#347](https://github.com/mgd43b/manicule/issues/347)) ([c420935](https://github.com/mgd43b/manicule/commit/c420935f7aef8fa1600d81db2295c7a2328524f9))
* **tests:** migrate the database once and copy it, rather than per test ([#351](https://github.com/mgd43b/manicule/issues/351)) ([3724d91](https://github.com/mgd43b/manicule/commit/3724d912b84490122679f5f9da4f3371726cc409))

## [0.1.18](https://github.com/mgd43b/manicule/compare/v0.1.17...v0.1.18) (2026-09-13)


### Features

* **storage:** serve the vector index from a Qdrant server ([#346](https://github.com/mgd43b/manicule/issues/346)) ([393f615](https://github.com/mgd43b/manicule/commit/393f615398183aaa1f9ae711b2501dacf72be4ce))

## [0.1.17](https://github.com/mgd43b/manicule/compare/v0.1.16...v0.1.17) (2026-09-13)


### Features

* **container:** publish the image to ghcr.io on release ([#338](https://github.com/mgd43b/manicule/issues/338)) ([88360d2](https://github.com/mgd43b/manicule/commit/88360d2f0d9ffe66fc6b276536a4b8e09ee0786d))

## [0.1.16](https://github.com/mgd43b/manicule/compare/v0.1.15...v0.1.16) (2026-09-13)


### Features

* author documents into the corpus, and extract [[wikilinks]] as relations ([#336](https://github.com/mgd43b/manicule/issues/336)) ([67f1da5](https://github.com/mgd43b/manicule/commit/67f1da51239930944939de78d6bce0e6afb49622))

## [0.1.15](https://github.com/mgd43b/manicule/compare/v0.1.14...v0.1.15) (2026-09-01)


### Features

* answer a question from several searches with `research` ([#326](https://github.com/mgd43b/manicule/issues/326)) ([fc0d702](https://github.com/mgd43b/manicule/commit/fc0d702f79d8dd0c4fd58659afec13b99e633a2d))


### Bug Fixes

* follow the grammar pack when a configured cache stopped being the library directory ([#328](https://github.com/mgd43b/manicule/issues/328)) ([e594a99](https://github.com/mgd43b/manicule/commit/e594a99f60fa783e049e7d39454ca3fbb37aa256))
* stop a policy drop silently changing the context's tokenizer ([#325](https://github.com/mgd43b/manicule/issues/325)) ([21e9114](https://github.com/mgd43b/manicule/commit/21e9114c76bc0349509ffde157cb64d0f87526ea))
* stop a takeover test racing the machine it runs on ([#329](https://github.com/mgd43b/manicule/issues/329)) ([4a312fb](https://github.com/mgd43b/manicule/commit/4a312fb6e2cbf189af1eb23ff53acfa4b73b320c))
* use a renewed browser session without a restart, and install the host for the right workspace ([#330](https://github.com/mgd43b/manicule/issues/330)) ([0f36215](https://github.com/mgd43b/manicule/commit/0f36215a175049d1e183f018a2e835452ba3d45b))

## [0.1.14](https://github.com/mgd43b/manicule/compare/v0.1.13...v0.1.14) (2026-08-27)


### Features

* **api:** resolve a cached document by page id, URI or document id ([#318](https://github.com/mgd43b/manicule/issues/318)) ([46b2338](https://github.com/mgd43b/manicule/commit/46b233830ff83de0e74678a9b17d850f5828f056))

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
