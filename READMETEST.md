# 1) env + deps
conda create -n struq python=3.10 -y && conda activate struq
pip install -r requirements.txt
pip install sentence-transformers umap-learn matplotlib

# 2) get data
python setup.py

# 3) build clusters + gate (writes security/frontier.pkl)
python scripts/frontier_quickstart.py

# 4) quick test a base model against attacks (downloads from HF)
python test.py -m huggyllama/llama-7b -a completion_real
