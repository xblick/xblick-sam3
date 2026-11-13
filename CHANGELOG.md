# Changelog

## 0.2.0 (2026-11-13)

Every name in code, on the wire and on disk is `xblick` now. Only the prose name of the project stays Durchblick.
The node needs the core `xblick` 0.5 and does not work with programs of Durchblick 0.4 (`blick`) or older: run every
program of a setup at 0.5.

| What | 0.1.0 | 0.2.0 |
| --- | --- | --- |
| Repository | github.com/xblick/durchblick-sam3 | github.com/xblick/xblick-sam3 |
| Distribution | `durchblick-sam3` | `xblick-sam3` |
| Python package | `durchblick_sam3` | `xblick_sam3` |
| Commands | `durchblick-sam3`, `durchblick-sam3-prompts` | `xblick-sam3`, `xblick-sam3-prompts` |
| Program name in log lines, the logger and reports | `durchblick-sam3` | `xblick-sam3` |
| Core dependency | `blick>=0.4,<0.5`, git main of github.com/xblick/durchblick | `xblick>=0.5.0,<0.6`, github.com/xblick/xblick at 45908e6 |
| Core imports and command | `import blick`, `blick rerun`, `blick play`, `blick param` | `import xblick`, `xblick rerun`, `xblick play`, `xblick param` |
| Links to the core's docs | github.com/xblick/durchblick/blob/main/docs/... | github.com/xblick/xblick/blob/main/docs/... |
| Other nodes named in the docs | `durchblick-foundationpose`, `durchblick-tapnext`, `durchblick-mapping` | `xblick-foundationpose`, `xblick-tapnext`, `xblick-mapping` |

## 0.1.0

The node as `durchblick-sam3` (Python package `durchblick_sam3`) on the core `blick` 0.4.
