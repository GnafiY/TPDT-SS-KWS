## Overview

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![arXiv](https://img.shields.io/badge/arXiv-2406.12447-b31b1b.svg)](https://arxiv.org/abs/2406.12447)

This is the official repository for Interspeech 2024 paper [*Text-aware Speech Separation for Multi-talker Keyword Spotting*](https://www.isca-archive.org/interspeech_2024/li24r_interspeech.pdf). The implementaion of the front-end model is based on [*ESPnet*](https://github.com/espnet/espnet). All unused examples in <code>egs</code> and <code>egs2</code> are removed. As for the KWS backend, We directly apply the default setup of MDTC from [*WeKws*](https://github.com/wenet-e2e/wekws) <code>examples/hey_snips/s0</code>.

I apologize that the email address of the primary author is wrong, which should be *haoyu.li.cs@sjtu.edu.cn* instead of *haoyu.li@sjtu.edu.cn*. Feel free to [mail to me](mailto:haoyu.li.cs@sjtu.edu.cn) if you have any question!

## Setup

1. Clone this repository.
2. Install ESPnet dependencies, please refer to [*ESPnet*](https://github.com/espnet/espnet) official repository.
3. Change directory to <code>espnet/egs2/librimix/enh1</code>.
4. Generate Libri2Mix scp data by running <code>bash run.sh --stage 1 --stop_stage 4</code>.
5. Generate Snips2Mix data with instruction in <code>local/Snips2Mix</code>.
6. Train and run inference by <code>bash run.sh --stage 5 --stop_stage 6</code> and <code>bash run.sh --stage 7 --stop_stage 8</code>, respectively.
7. If you wish to run KWS inference, please refer to the snips recipe in [*WeKws*](https://github.com/wenet-e2e/wekws).