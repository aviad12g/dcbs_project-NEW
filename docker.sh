docker stop dcbs
docker rm dcbs
docker build --rm -t dcbs .
docker run -d -v $HOME/.huggingface:/root/.huggingface -v $HOME/.cache/huggingface:/root/.cache/huggingface --gpus '"device=0,1,2,3"' --name dcbs dcbs
docker logs -f dcbs