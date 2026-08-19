# Changelog

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
