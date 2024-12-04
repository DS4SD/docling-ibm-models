## [v2.0.7](https://github.com/DS4SD/docling-ibm-models/releases/tag/v2.0.7) - 2024-12-02

### Fix

* Improve numpy compatibility pinning ([#57](https://github.com/DS4SD/docling-ibm-models/issues/57)) ([`de2f241`](https://github.com/DS4SD/docling-ibm-models/commit/de2f241ea8577636bd72367a97691613e93e20de))

## [v2.0.6](https://github.com/DS4SD/docling-ibm-models/releases/tag/v2.0.6) - 2024-11-20

### Fix

* Python3.9 support ([#54](https://github.com/DS4SD/docling-ibm-models/issues/54)) ([`e2b19d9`](https://github.com/DS4SD/docling-ibm-models/commit/e2b19d930279c150557af33f2b08d7abb1f47428))

## [v2.0.5](https://github.com/DS4SD/docling-ibm-models/releases/tag/v2.0.5) - 2024-11-20

### Fix

* Removing dependency from mean_average_precision package (not in use) ([#53](https://github.com/DS4SD/docling-ibm-models/issues/53)) ([`65affef`](https://github.com/DS4SD/docling-ibm-models/commit/65affef1f3ff209a8cdae4201aaaad8872d1069b))

## [v2.0.4](https://github.com/DS4SD/docling-ibm-models/releases/tag/v2.0.4) - 2024-11-18

### Fix

* Remove lxml deps ([#51](https://github.com/DS4SD/docling-ibm-models/issues/51)) ([`7a0cbde`](https://github.com/DS4SD/docling-ibm-models/commit/7a0cbde7e0638bbd91d5905ae457fdcb299ee87f))

## [v2.0.3](https://github.com/DS4SD/docling-ibm-models/releases/tag/v2.0.3) - 2024-10-30

### Fix

* Simplify torch dependencies in the wheels ([#45](https://github.com/DS4SD/docling-ibm-models/issues/45)) ([`bca09f8`](https://github.com/DS4SD/docling-ibm-models/commit/bca09f8331934120e6f542d15aa871f1153ae140))

## [v2.0.2](https://github.com/DS4SD/docling-ibm-models/releases/tag/v2.0.2) - 2024-10-29

### Fix

* **LayoutPredictor:** Ensure that the predicted bboxes are minmaxed inside the image boundaries ([#42](https://github.com/DS4SD/docling-ibm-models/issues/42)) ([`216cee0`](https://github.com/DS4SD/docling-ibm-models/commit/216cee081a1ae8be264074e6d54cff27bf6984cf))

## [v2.0.1](https://github.com/DS4SD/docling-ibm-models/releases/tag/v2.0.1) - 2024-10-16

### Fix

* Numpy with python 3.13 support ([#39](https://github.com/DS4SD/docling-ibm-models/issues/39)) ([`4fddc45`](https://github.com/DS4SD/docling-ibm-models/commit/4fddc45cd7b7684a28b5556cbf0681540d5d4010))

## [v2.0.0](https://github.com/DS4SD/docling-ibm-models/releases/tag/v2.0.0) - 2024-10-03

### Feature

* Release v2.0.0 with only torch models ([#38](https://github.com/DS4SD/docling-ibm-models/issues/38)) ([`8719555`](https://github.com/DS4SD/docling-ibm-models/commit/8719555d661dde491667e76d3de6abcd3d1b25bd))

### Breaking

* release v2.0.0 with only torch models ([#38](https://github.com/DS4SD/docling-ibm-models/issues/38)) ([`8719555`](https://github.com/DS4SD/docling-ibm-models/commit/8719555d661dde491667e76d3de6abcd3d1b25bd))

## [v1.4.0](https://github.com/DS4SD/docling-ibm-models/releases/tag/v1.4.0) - 2024-10-03

### Feature

* Migration from onnx to pytorch script ([#37](https://github.com/DS4SD/docling-ibm-models/issues/37)) ([`59e2941`](https://github.com/DS4SD/docling-ibm-models/commit/59e2941f41e4d419ab650d58e7ca1a667ae31002))

## [v1.3.3](https://github.com/DS4SD/docling-ibm-models/releases/tag/v1.3.3) - 2024-10-01

### Fix

* Put back in common.py the function `read_config()`. Extend the unit tests. ([#36](https://github.com/DS4SD/docling-ibm-models/issues/36)) ([`d0bdb22`](https://github.com/DS4SD/docling-ibm-models/commit/d0bdb22b1535bfe118e7ec88ea9f2b17ca4469e8))

## [v1.3.2](https://github.com/DS4SD/docling-ibm-models/releases/tag/v1.3.2) - 2024-09-30

### Fix

* Remove left-over code which is not needed for prediction ([#35](https://github.com/DS4SD/docling-ibm-models/issues/35)) ([`b6ba0c7`](https://github.com/DS4SD/docling-ibm-models/commit/b6ba0c7749177eb25c3ef79188f2508be211150a))

## [v1.3.1](https://github.com/DS4SD/docling-ibm-models/releases/tag/v1.3.1) - 2024-09-27

### Fix

* Pinned opencv-python-headless to version "4.6.0.66" ([#34](https://github.com/DS4SD/docling-ibm-models/issues/34)) ([`484340f`](https://github.com/DS4SD/docling-ibm-models/commit/484340fd1a3a7a31babaf2306496c083b56e4c50))

## [v1.3.0](https://github.com/DS4SD/docling-ibm-models/releases/tag/v1.3.0) - 2024-09-23

### Feature

* Extend the tests and demo to first download the model files from HF. Add the pytest in GitHub workflow ( #30) ([`79888d0`](https://github.com/DS4SD/docling-ibm-models/commit/79888d0b87172af43f2585b54e974e54130bab4e))

## [v1.2.1](https://github.com/DS4SD/docling-ibm-models/releases/tag/v1.2.1) - 2024-09-18

### Fix

* Safer bbox processing ([#27](https://github.com/DS4SD/docling-ibm-models/issues/27)) ([`d37272e`](https://github.com/DS4SD/docling-ibm-models/commit/d37272eb8e441045dc7c903530db4d1afd051f3e))
* Make col/row re-sorting optional on TF predictor (#19) [skip ci] ([`7e9758c`](https://github.com/DS4SD/docling-ibm-models/commit/7e9758c684317143ffd7bbfba71ad8d8328854b7))

## [v1.2.0](https://github.com/DS4SD/docling-ibm-models/releases/tag/v1.2.0) - 2024-09-17

### Feature

* **LayoutPredictor:** Introduce black-listed classes which are filtered out from the response. ([#26](https://github.com/DS4SD/docling-ibm-models/issues/26)) ([`86a6a50`](https://github.com/DS4SD/docling-ibm-models/commit/86a6a5071bd84dd2b6806ac413a3163e78b6127f))

## [v1.1.7](https://github.com/DS4SD/docling-ibm-models/releases/tag/v1.1.7) - 2024-09-05

### Fix

* Validation and typechecks in TF post processing and OTSL to HTML conversion function ([#18](https://github.com/DS4SD/docling-ibm-models/issues/18)) ([`d607914`](https://github.com/DS4SD/docling-ibm-models/commit/d60791489596c4052b853d3c1dc559ba026a9f77))

## [v1.1.6](https://github.com/DS4SD/docling-ibm-models/releases/tag/v1.1.6) - 2024-09-03

### Fix

* TableFormer raises IndexError: too many indices for array ([`ad494ca`](https://github.com/DS4SD/docling-ibm-models/commit/ad494caaaceb46fcd99054dba70c10fc688a57e6))

## [v1.1.5](https://github.com/DS4SD/docling-ibm-models/releases/tag/v1.1.5) - 2024-08-29

### Fix

* **poetry:** Remove unused dependencies from toml. Update lock. ([#16](https://github.com/DS4SD/docling-ibm-models/issues/16)) ([`3792577`](https://github.com/DS4SD/docling-ibm-models/commit/379257780f12a26770a93a0e3dbf61e36531affc))

## [v1.1.4](https://github.com/DS4SD/docling-ibm-models/releases/tag/v1.1.4) - 2024-08-28

### Fix

* Fix torch dependency for Intel Macs ([#15](https://github.com/DS4SD/docling-ibm-models/issues/15)) ([`12153f8`](https://github.com/DS4SD/docling-ibm-models/commit/12153f829f2ccf25ab03a12924982b1220bf9d6e))

## [v1.1.3](https://github.com/DS4SD/docling-ibm-models/releases/tag/v1.1.3) - 2024-08-27

### Fix

* Table cell overlap removal in TF post-processing: ([#10](https://github.com/DS4SD/docling-ibm-models/issues/10)) ([`e8f396d`](https://github.com/DS4SD/docling-ibm-models/commit/e8f396d19a277b0f3147c3ac3af0b5e452912879))

## [v1.1.2](https://github.com/DS4SD/docling-ibm-models/releases/tag/v1.1.2) - 2024-08-21

### Fix

* Align to use opencv-python-headless ([#12](https://github.com/DS4SD/docling-ibm-models/issues/12)) ([`22097fb`](https://github.com/DS4SD/docling-ibm-models/commit/22097fb3a451c9a2e4531666f96b4a126701b3b4))

## [v1.1.1](https://github.com/DS4SD/docling-ibm-models/releases/tag/v1.1.1) - 2024-08-14

### Fix

* Allow newer torch and update deps ([#9](https://github.com/DS4SD/docling-ibm-models/issues/9)) ([`79e389a`](https://github.com/DS4SD/docling-ibm-models/commit/79e389a60f2fb9295617198d6985957aab53f04d))

## [v1.1.0](https://github.com/DS4SD/docling-ibm-models/releases/tag/v1.1.0) - 2024-07-18

### Feature

* Switch to python3.10. Update poetry files, readme.md, contributing.md ([`b9efd29`](https://github.com/DS4SD/docling-ibm-models/commit/b9efd29dae7de59547be9eca457a227b871fa03e))

### Documentation

* Optimize images ([#6](https://github.com/DS4SD/docling-ibm-models/issues/6)) ([`cbfcd4e`](https://github.com/DS4SD/docling-ibm-models/commit/cbfcd4e9f2f9ba840eec034045c8b410ad83df44))

## [v1.0.0](https://github.com/DS4SD/docling-ibm-models/releases/tag/v1.0.0) - 2024-07-16

### Feature

* First ci release ([#5](https://github.com/DS4SD/docling-ibm-models/issues/5)) ([`1c45cb5`](https://github.com/DS4SD/docling-ibm-models/commit/1c45cb5d81f88e2aaf99864b9df5d80b20d6ce94))

### Breaking

* first ci release ([#5](https://github.com/DS4SD/docling-ibm-models/issues/5)) ([`1c45cb5`](https://github.com/DS4SD/docling-ibm-models/commit/1c45cb5d81f88e2aaf99864b9df5d80b20d6ce94))

