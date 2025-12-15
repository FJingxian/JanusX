python -m pip config set global.index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple # set mirror
python -m pip install --upgrade pip
python -m pip install uv
python -m uv venv --clear
python -m uv sync
chmod +x jx