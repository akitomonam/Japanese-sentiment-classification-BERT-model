#!/bin/sh

# usage: ./set_dot_env "proxy_url"

PROXY=$1
touch .env

echo "PROXY=$1" >> .env

echo "UID=$(id -u $USER)" >> .env
echo "GID=$(id -g $USER)" >> .env
echo "UNAME=$USER" >> .env
echo "WANDB_API_KEY=" >> .env