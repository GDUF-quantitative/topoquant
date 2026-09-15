行情数据目录（data/stock）
==========================

本目录用于放置个股原始行情 CSV，是流水线运行的必需输入。

放置要求：
- 每个股票一个 CSV 文件，命名形如 `000001.SZ.csv`、`600000.SH.csv`
  （前缀 6 位代码 + 交易所后缀，与源项目 data/stock 命名一致）。
- 文件需包含列：EventDate、open、high、low、close、volume、money、prev_close
  等（详见 src/topoquant/data.py 的字段解析）。
- 放入后运行 setup_env.py 会自动检测并提示数据就绪。

注意：本纯净版默认【不含】行情数据，请自行将原始 data/stock 内容复制到此目录，
或软链/挂载到此处，然后执行 `python setup_env.py` 完成环境初始化。
