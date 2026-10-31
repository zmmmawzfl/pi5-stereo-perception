PY ?= .venv/bin/python
CALIB ?= runs/demo/calib/stereo_calib.npz

.PHONY: help test demo calib day1 clean

help:
	@echo "make test   运行 61 个测试"
	@echo "make demo   合成场景端到端 demo + 压缩量化"
	@echo "make calib  生成合成标定文件"
	@echo "make day1   Pi 端时间戳基线（无相机时自动用 synthetic）"

test:
	$(PY) -m unittest discover -s tests -t . -v

demo:
	$(PY) -m tools.run_demo --out runs/demo --frames 8 --compression

calib:
	$(PY) -m pc.calib.cli synthetic --out $(CALIB) --views 15

day1:
	$(PY) -m pi.tools.timestamp_baseline --frames 100 --out runs/day1_baseline.json

clean:
	rm -rf runs/replay runs/live runs/compression
