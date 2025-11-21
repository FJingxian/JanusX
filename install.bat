python -m pip config set global.index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
python -m pip install uv
python -m uv venv --clear
python -m uv cache clean
python -m uv sync
copy doc\windows.bat gtools.bat