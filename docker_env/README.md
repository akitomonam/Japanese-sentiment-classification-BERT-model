# dockerを使った環境構築
# 方法
## 環境変数の設定
```
./set_dot_env.sh "proxy_url"
```
##コンテナ構築
```
docker-compose up -d
```
コンテナ内にアクセス
```
docker-compose exec dialogue-compe-env bash
```
アクセス後いろいろインストールしてよし。
```
pip install pytorch
```
など。

※DockerコマンドでpermissionErrorと出たら、先頭にsudoをつける
