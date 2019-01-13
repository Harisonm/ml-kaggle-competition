<h1>Cheat sheet</h1>

### docker-compose
```
# Builde Docker-compose
docker-compose build

# Up Docker
docker-compose up -d

# Show logs of docker-compose
docker-compose logs

# show procesus status docker-compose
docker-compose ps
```

<h1>Documentations</h1>

# Anaconda 3 docker
1. Depuis votre repertoire de travail -> Lancer l'environnement python36 d'anaconda
```
docker run -it -v $(pwd):path_dir -p 8888:8888 sources_notebook /bin/bash -c "jupyter-notebook --notebook-dir=path_folder_of_computer --no-browser --port=8888 --ip=0.0.0.0 --allow-root "
```

1.bis  Run shell command for notebook on start

``` 
jupyter-notebook --notebook-dir=path_dir --no-browser --port=8888 --ip=0.0.0.0 --allow-root &
```

### Si vous voulez ajouter d'autre environnement Python Anaconda dans Jupyter

```
python -m ipykernel install --user --name envPython36 --display-name "Python (envPython36)"
```

# Containers Python

### lancer le python36 (hors anaconda)
```
docker exec -it DEFAULT_PYTHON sh -c "cd /srv/python; sh"
```

### Docker Containers
```
# list
docker ps

# stop all 
docker stop $(docker ps -q)

# remove all
docker rm $(docker ps -a -q)

# stop & remove all
docker stop $(docker ps -a -q) && docker rm $(docker ps -a -q)
```



