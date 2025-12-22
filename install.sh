python -m pip config set global.index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple # set mirror
python -m pip install --upgrade pip
python -m pip install uv
python -m uv venv --clear
python -m uv sync
uv pip install -e ./ext/glm_rs
cp ./doc/unix.sh jx
./jx -h
echo "Recommend: Add $(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd) to PATH"