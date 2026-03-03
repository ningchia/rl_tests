## Python 套件安裝說明

請先安裝 Python 3.9+，再依需求安裝以下套件。

### 一次安裝（建議）
> 若你要同時執行 `01_rl_frozen_lake.py` 與 `02_rl_cart_pole.py`

```bash
pip install -U numpy gymnasium[toy-text] torch tensorboard "setuptools<82.0.0"
```

### 依腳本安裝

**01_rl_frozen_lake.py**
```bash
pip install -U numpy "gymnasium[toy-text]"
```

**02_rl_cart_pole.py**
```bash
pip install -U numpy gymnasium torch tensorboard "setuptools<82.0.0"
```

### 備註
- `tensorboard` 需要 `setuptools<82.0.0`，以避免 `pkg_resources` 模組被移除造成的錯誤。
- 如需啟動 TensorBoard，請在專案目錄執行：
```bash
tensorboard --logdir=runs
```
- 接下來用瀏覽器開啟 http://localhost:6006
